import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns; sns.set()
import glob2
import argparse


def smooth_reward_curve(x, y, padding, a_range=20000):
    if a_range != 20000:
        len_m = a_range
    else:
        len_m = len(x)
    halfwidth = int(np.ceil(len_m / 100))
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(padding * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(padding * k + 1),
        mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result


def pad(xs, value=np.nan):
    maxlen = np.max([len(x) for x in xs])
    max_index = 0
    for index, x in enumerate(xs):
        if len(x) == maxlen:
            max_index = index
    
    padded_xs = []
    index=-1
    for x in xs:
        if x.shape[0] >= maxlen:
            padded_xs.append(x)
            index += 1
        else:
            padding = np.ones((maxlen - x.shape[0],) + x.shape[1:]) * value
            x_padded = np.concatenate([x, padding], axis=0)
            assert x_padded.shape[1:] == x.shape[1:]
            assert x_padded.shape[0] == maxlen
            padded_xs.append(x_padded)
    return np.array(padded_xs), index, max_index


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='')
parser.add_argument('--title', type=str, default='')
parser.add_argument('--type', type=str, default='test')
parser.add_argument('--dir1', type=str, default=0)
parser.add_argument('--dir2', type=str, default=0)
parser.add_argument('--dir3', type=str, default=0)
parser.add_argument('--dir4', type=str, default=0)
parser.add_argument('--dir5', type=str, default=0)
parser.add_argument('--dir6', type=str, default=0)
parser.add_argument('--dir7', type=str, default=0)
parser.add_argument('--dir8', type=str, default=0)
parser.add_argument('--dir9', type=str, default=0)
parser.add_argument('--dir10', type=str, default=0)
parser.add_argument('--dir11', type=str, default=0)
parser.add_argument('--dir12', type=str, default=0)
parser.add_argument('--dir13', type=str, default=0)
parser.add_argument('--dir14', type=str, default=0)
parser.add_argument('--dir15', type=str, default=0)
parser.add_argument('--dir16', type=str, default=0)
parser.add_argument('--dir17', type=str, default=0)
parser.add_argument('--dir18', type=str, default=0)
parser.add_argument('--dir19', type=str, default=0)
parser.add_argument('--dir20', type=str, default=0)
parser.add_argument('--dir21', type=str, default=0)
parser.add_argument('--dir22', type=str, default=0)
parser.add_argument('--dir23', type=str, default=0)
parser.add_argument('--dir24', type=str, default=0)
parser.add_argument('--dir25', type=str, default=0)
parser.add_argument('--dir26', type=str, default=0)
parser.add_argument('--dir27', type=str, default=0)
parser.add_argument('--dir28', type=str, default=0)
parser.add_argument('--dir29', type=str, default=0)
parser.add_argument('--dir30', type=str, default=0)
parser.add_argument('--dir31', type=str, default=0)
parser.add_argument('--dir32', type=str, default=0)
parser.add_argument('--dir33', type=str, default=0)
parser.add_argument('--dir34', type=str, default=0)
parser.add_argument('--dir35', type=str, default=0)
parser.add_argument('--dir36', type=str, default=0)
parser.add_argument('--dir37', type=str, default=0)
parser.add_argument('--dir38', type=str, default=0)
parser.add_argument('--dir39', type=str, default=0)
parser.add_argument('--dir40', type=str, default=0)
parser.add_argument('--dir41', type=str, default=0)
parser.add_argument('--dir42', type=str, default=0)
parser.add_argument('--dir43', type=str, default=0)
parser.add_argument('--dir44', type=str, default=0)
parser.add_argument('--dir45', type=str, default=0)
parser.add_argument('--dir46', type=str, default=0)
parser.add_argument('--dir47', type=str, default=0)
parser.add_argument('--dir48', type=str, default=0)
parser.add_argument('--dir49', type=str, default=0)
parser.add_argument('--dir50', type=str, default=0)
parser.add_argument('--upper_reward', type=str, default=0)
parser.add_argument('--lower_reward', type=str, default=0)
parser.add_argument('--padding', type=int, default=10)
parser.add_argument('--plot_name', type=str, default='')
parser.add_argument('--smooth', type=int, default=1)
parser.add_argument('--range', type=int, default=20000)

args = parser.parse_args()
path_list = []

path_list_dict = dict()
dir_list_dict = dict()

if args.dir1:
    dir_list_dict['dir1'] = args.dir1
if args.dir2:
    dir_list_dict['dir2'] = args.dir2
if args.dir3:
    dir_list_dict['dir3'] = args.dir3
if args.dir4:
    dir_list_dict['dir4'] = args.dir4
if args.dir5:
    dir_list_dict['dir5'] = args.dir5
if args.dir6:
    dir_list_dict['dir6'] = args.dir6
if args.dir7:
    dir_list_dict['dir7'] = args.dir7
if args.dir8:
    dir_list_dict['dir8'] = args.dir8
if args.dir9:
    dir_list_dict['dir9'] = args.dir9
if args.dir10:
    dir_list_dict['dir10'] = args.dir10
if args.dir11:
    dir_list_dict['dir11'] = args.dir11
if args.dir12:
    dir_list_dict['dir12'] = args.dir12
if args.dir13:
    dir_list_dict['dir13'] = args.dir13
if args.dir14:
    dir_list_dict['dir14'] = args.dir14
if args.dir15:
    dir_list_dict['dir15'] = args.dir15
if args.dir16:
    dir_list_dict['dir16'] = args.dir16
if args.dir17:
    dir_list_dict['dir17'] = args.dir17
if args.dir18:
    dir_list_dict['dir18'] = args.dir18
if args.dir19:
    dir_list_dict['dir19'] = args.dir19
if args.dir20:
    dir_list_dict['dir20'] = args.dir20
if args.dir21:
    dir_list_dict['dir21'] = args.dir21
if args.dir22:
    dir_list_dict['dir22'] = args.dir22
if args.dir23:
    dir_list_dict['dir23'] = args.dir23
if args.dir24:
    dir_list_dict['dir24'] = args.dir24
if args.dir25:
    dir_list_dict['dir25'] = args.dir25
if args.dir26:
    dir_list_dict['dir26'] = args.dir26
if args.dir27:
    dir_list_dict['dir27'] = args.dir27
if args.dir28:
    dir_list_dict['dir28'] = args.dir28
if args.dir29:
    dir_list_dict['dir29'] = args.dir29
if args.dir30:
    dir_list_dict['dir30'] = args.dir30
if args.dir31:
    dir_list_dict['dir31'] = args.dir31
if args.dir32:
    dir_list_dict['dir32'] = args.dir32
if args.dir33:
    dir_list_dict['dir33'] = args.dir33
if args.dir34:
    dir_list_dict['dir34'] = args.dir34
if args.dir35:
    dir_list_dict['dir35'] = args.dir35
if args.dir36:
    dir_list_dict['dir36'] = args.dir36
if args.dir37:
    dir_list_dict['dir37'] = args.dir37
if args.dir38:
    dir_list_dict['dir38'] = args.dir38
if args.dir39:
    dir_list_dict['dir39'] = args.dir39
if args.dir40:
    dir_list_dict['dir40'] = args.dir40
if args.dir41:
    dir_list_dict['dir41'] = args.dir41
if args.dir42:
    dir_list_dict['dir42'] = args.dir42
if args.dir43:
    dir_list_dict['dir43'] = args.dir43
if args.dir44:
    dir_list_dict['dir44'] = args.dir44
if args.dir45:
    dir_list_dict['dir45'] = args.dir45
if args.dir46:
    dir_list_dict['dir46'] = args.dir46
if args.dir47:
    dir_list_dict['dir47'] = args.dir47
if args.dir48:
    dir_list_dict['dir48'] = args.dir48
if args.dir49:
    dir_list_dict['dir49'] = args.dir49
if args.dir50:
    dir_list_dict['dir50'] = args.dir50

# Load all data.
data = {}
for i in range(len(dir_list_dict)):
    i += 1
    if dir_list_dict['dir'+str(i)] != '':
        dir_list_dict['dir'+str(i)] = './models' + args.dir + '/' + dir_list_dict['dir'+str(i)]

# print(dir_list_dict)
        
for i in range(len(dir_list_dict)):
    curr_path = dir_list_dict['dir'+str(i+1)].split(':')[0]
    if curr_path == '':
        continue
    if not os.path.isdir(curr_path):
        print("Not os path is current path")
        continue
    results = load_results(os.path.join(curr_path, 'progress.csv'))
    if not results:
        print('skipping {}'.format(curr_path))
        continue
    print('loading {} ({})'.format(curr_path, len(results['1.  Epoch'])))
    with open(os.path.join(curr_path, 'params.json'), 'r') as f:
        params = json.load(f) #load the parameters from the param file
    
    # if args.type == 'train':
    #     success_rate = np.array(results['5.  Train success rate'])
    # else:
    #     success_rate = np.array(results['9.  Test success rate'])
    if args.upper_reward:
        if args.type == 'train':
            success_rate = np.array(results['6.  Train average upper reward'])
        else:
            success_rate = np.array(results['X.  Test average upper reward'])

    elif args.lower_reward:
        if args.type == 'train':
            success_rate = np.array(results['7.  Train average lower reward'])
        else:
            success_rate = np.array(results['XI. Test average lower reward'])

    else:
        if args.type == 'train':
            success_rate = np.array(results['5.  Train success rate'])
        else:
            success_rate = np.array(results['9.  Test success rate'])
    epoch = np.array(results['2.  Timesteps']) + 1
    env_id = params['env_name']
    replay_strategy = params['replay_strategy']

    iden = dir_list_dict['dir'+str(i+1)].split(':')[-1]
    env_id += iden
    config = iden

    # Process and smooth data.
    assert success_rate.shape == epoch.shape
    x = epoch
    y = success_rate
    if args.smooth:
        x, y = smooth_reward_curve(epoch, success_rate, args.padding, args.range) #smoothen the data
    assert x.shape == y.shape

    z=y
    if env_id not in data:
        data[env_id] = {}
    if config not in data[env_id]:
        data[env_id][config] = []
    data[env_id][config].append((x, y, z))

# Plot data.
plt.clf()
plt.figure(1)
plt.xlabel('Epoch')
plt.ylabel('Success Rate')
plt.title(args.title, fontsize=12)
plt.legend()
i = -1
# colours = ['black', 'r', 'deepskyblue', 'g', 'y', 'hotpink', 'saddlebrown', 'blue', 'gray', 'gold', 'seagreen', 'limegreen']
# colours = ['red', 'lawngreen', 'green', 'y', 'k', 'c','m', 'aquamarine', 'coral', 'mediumblue', 'gold', 'seagreen']

colours = ['black', 'r', 'deepskyblue', 'limegreen', 'g', 'hotpink', 'saddlebrown', 'blue', 'gray', 'darkorange', 'lawngreen', 'cyan','blueviolet', 'darkmagenta', 'darkblue', 'pink', 'lavenderblush']
# sorted_keys = np.array(['PEAR-IRL', 'PEAR-BC', 'RPL', 'HAC', 'RAPS', 'HIER-NEG', 'HIER', 'DAC', 'FLAT'])
sorted_keys = np.array(['CRISP-IRL', 'CRISP-BC', 'RPL', 'HAC', 'RAPS', 'HIER-NEG', 'HIER', 'DAC', 'FLAT'])
sorted_keys_2 = []
for j in range(len(sorted_keys)):
    # if sorted_keys[j] != 'BC':
    sorted_keys_2.append(params['env_name'] + sorted_keys[j])
# start_flag = 0


for env_id in sorted(data.keys()): # for all the env ids
    i += 1

    for config in sorted(data[env_id].keys()): #for all the configs in env ids
        xs, ys, zs = zip(*data[env_id][config])
        xs, _, index = pad(xs)
        ys,_,_ = pad(ys)
        zs,_,_ = pad(zs)
        assert xs.shape == ys.shape
        assert xs.shape == zs.shape
        # if args.range:
        xs = xs[:,:args.range]
        ys = ys[:,:args.range]
        zs = zs[:,:args.range]
        
        plt.plot(xs[index], np.nanmedian(ys, axis=0), color=colours[i], label=config)
        plt.fill_between(xs[index], np.nanpercentile(ys, 25, axis=0), np.nanpercentile(ys, 75, axis=0), alpha=0.25, color=colours[i])
        plt.xlabel('Timesteps', fontsize=12)
        plt.ylabel('Success Rate', fontsize=12)
        current_values = plt.gca().get_xticks()
        plt.gca().set_xticklabels(['{:.1f}'.format(x/1000000.)+'M' if x >= 1000000 else '{:.0f}'.format(x/1000)+'k' if x >= 1000 else '{:.0f}'.format(x) for x in current_values])

        plt.legend(fontsize=11)

plt.grid(b=None)
plt.subplots_adjust(top=0.92, bottom=0.1, left=0.15, right=0.95, hspace=0.3, wspace=0.35)

temp_name = '_success_rate'

temp_name = '_success_rate'
if args.upper_reward:
    temp_name = '_upper_reward'
if args.lower_reward:
    temp_name = '_lower_reward'

plt.savefig(os.path.join('./figs/', '{}.png'.format(args.plot_name+ temp_name)))
plt.show()
        


    
    


    
