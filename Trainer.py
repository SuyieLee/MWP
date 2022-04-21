import torch.optim as optim
from torch.nn import Dropout
from torch.autograd import Variable
from data_tools import post_solver, inverse_temp_to_num
import torch
import torch.nn as nn
import os
from data_tools import out_expression_list, compute_prefix_expression
import copy


class Trainer(object):
    def __init__(self, model, loss=None, weight=None, vocab_dict=None, vocab_list=None, data_loader=None, batch_size=32, decode_classes_dict=None, decode_classes_list=None,
                 cuda_use=True, print_every=10, checkpoint_dir_name=None):
        self.model = model
        self.vocab_dict = vocab_dict
        self.vocab_list = vocab_list
        self.data_loader = data_loader
        self.decode_classes_dict = decode_classes_dict
        self.decode_classes_list = decode_classes_list
        self.cuda_use = cuda_use
        self.print_every = print_every
        # self.optimizer = optim.Adam(model.parameters())
        self.batch_size = batch_size
        self.MAX_OUTPUT_LENGTH = 45
        if loss is None:
            self.criterion = nn.NLLLoss(weight=weight, reduction='mean')
        else:
            self.criterion = loss

    def train(self, model, epoch_num=100, start_epoch=0, valid_every=10):
        train_list = self.data_loader.train_data
        valid_list = self.data_loader.valid_data
        best_valid = 0
        path = ""

        for epoch in range(start_epoch, epoch_num):
            model.encoder_scheduler.step()
            model.prediction_scheduler.step()
            model.generation_scheduler.step()
            model.merge_scheduler.step()
            
            start_step = 0
            total_num = 0
            total_loss = 0
            total_acc_num = 0
            print("Epoch " + str(epoch+1) + " start training!")
            for batch in self.data_loader.yield_batch(train_list, self.batch_size):
                input = batch['batch_encode_pad_idx']
                input_len = batch['batch_encode_len']
                target = batch['batch_decode_pad_idx']
                target_len = batch['batch_decode_len']
                function_ans = batch['batch_ans']
                num_list = batch['batch_num_list']
                batch_num_count = batch['batch_num_count']
                batch_num_index_list = batch['batch_num_index_list']
                nums_stack_batch = batch['nums_stack_batch']

                total_num += len(input)

                loss = model(input, input_len, target, target_len, batch_num_count, self.data_loader.generate_op_index, batch_num_index_list, nums_stack_batch)
                total_loss += loss

                start_step += 1
                if start_step % self.print_every == 0:
                    print("Epoch %d Batch Loss: %.5f  |  Step %d Batch Train Loss: %.2f" % (epoch+1, total_loss/total_num, start_step, loss))

            if (epoch+1) % valid_every == 0:
                valid_acc = self.evaluate(model, valid_list)
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    path = os.path.join('./model/', "epoch_"+str(epoch+1)+"_result"+str(best_valid)+".pth")
                    torch.save(model.state_dict(), path)
                print("Best Valid Acc: %.2f | Valid Acc: %.2f" % (best_valid ,valid_acc))
        return path

    def evaluate(self, model, data):
        # Set to not-training mode to disable dropout
        model.encoder.eval()
        model.prediction.eval()
        model.generation.eval()
        model.merge.eval()
        total_loss = 0
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        for batch in self.data_loader.yield_batch(data, 1):
            input = batch['batch_encode_pad_idx'][0]
            input_len = batch['batch_encode_len'][0]
            target = batch['batch_decode_pad_idx'][0]
            target_len = batch['batch_decode_len'][0]
            function_ans = batch['batch_ans'][0]
            num_list = batch['batch_num_list'][0]
            batch_num_count = batch['batch_num_count'][0]
            batch_num_index_list = batch['batch_num_index_list'][0]
            nums_stack_batch = batch['nums_stack_batch'][0]

            test_res = model.test(input, input_len, self.data_loader.generate_op_index, batch_num_index_list, beam_size=5, max_length=self.MAX_OUTPUT_LENGTH)
            val_ac, equ_ac, _, _ = self.compute_prefix_tree_result(test_res, target, num_list, nums_stack_batch)
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        print(equation_ac, value_ac, eval_total)
        print("evaluate_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        return value_ac

    def compute_prefix_tree_result(self, test_res, test_tar, num_list, num_stack):
        # print(test_res, test_tar)

        if len(num_stack) == 0 and test_res == test_tar:
            return True, True, test_res, test_tar
        test = out_expression_list(test_res, self.data_loader, num_list)
        tar = out_expression_list(test_tar, self.data_loader, num_list, copy.deepcopy(num_stack))
        # print(test, tar)
        if test is None:
            return False, False, test, tar
        if test == tar:
            return True, True, test, tar
        try:
            if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
                return True, False, test, tar
            else:
                return False, False, test, tar
        except:
            return False, False, test, tar

    def get_ans_acc(self, output, function_ans, batch_size, num_list):
        acc = 0
        output = output.view(-1, batch_size, len(self.decode_classes_list))
        output = output.transpose(0, 1)
        for i in range(len(output)):
            templates = self.get_template(output[i])
            # print(templates)
            try:
                equ = inverse_temp_to_num(templates, num_list[i])
                # print(equ)
                predict_ans = post_solver(equ)
                if abs(float(predict_ans) - float(function_ans[i])) < 1e-5:
                    acc += 1
            except:
                acc += 0
        return acc

    def get_template(self, pred):
        templates = []
        for vec in pred:
            idx = vec.argmax(0).item()
            if idx == self.decode_classes_dict['PAD_token'] or idx == self.decode_classes_dict['END_token']:
                break
            templates.append(self.decode_classes_list[idx])
        return templates






