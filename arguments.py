# read and save the json file of training hyper-parameters
import os
import json

def readArguments(opt, parser, restore_var):
    flag_file = os.path.join(opt.resume, 'checkpoints', 'flags.json')
    if os.path.exists(flag_file):
        with open(flag_file, 'r') as f:
            stored_flags = {'--' + k: str(v) for k, v in json.load(f).items() if k in restore_var}
            to_del = []
            for flag, val in stored_flags.items():
                for act in parser._actions:
                    if act.dest == flag[2:]:
                        # store_true / store_false args don't accept arguments, filter these
                        if type(act.const) == type(True):
                            if val == str(act.default):
                                to_del.append(flag)
                            else:
                                stored_flags[flag] = ''
            for flag in to_del: del stored_flags[flag]

            train_flags = [x for x in list(sum(stored_flags.items(), tuple())) if len(x) > 0]
            print('Restored flags:', train_flags)
            opt = parser.parse_args(train_flags, namespace=opt)
    return opt