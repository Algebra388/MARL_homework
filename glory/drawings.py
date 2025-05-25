import matplotlib.pyplot as plt
import numpy as np
import ast

file_name_vdn = '2025-05-20 14_00_monster_lasthp_vdn_env_baseline.txt'

length = 4000

def smoothing(data, window_size = 20):
    smoothed_data = []
    for i in range(len(data)):
        start = max(0, i - window_size)
        end = min(len(data), i + window_size + 1)
        smoothed_data.append(np.mean(data[start: end]))
    return np.array(smoothed_data)

# Read the file and convert the string to a list
with open(file_name_vdn, 'r') as f:
    data = f.read()
    y_vdn_base = smoothing(np.array(ast.literal_eval(data))) # get list
    y_vdn_base = y_vdn_base[:length]
    x_vdn_base = np.array(range(len(y_vdn_base)))

file_name_qmix = '2025-05-22 14_27_monster_lasthp_qmix_env_baseline.txt'
with open(file_name_qmix, 'r') as f:
    data = f.read()
    y_qmix_base = smoothing(np.array(ast.literal_eval(data))) # get list
    y_qmix_base = y_qmix_base[:length]
    x_qmix_base = np.array(range(len(y_qmix_base)))


file_name_vdn_promotion = '2025-05-21 17_13_monster_lasthp_vdn_env_promotion.txt'
with open(file_name_vdn_promotion, 'r') as f:
    data = f.read()
    y_vdn_promotion = smoothing(np.array(ast.literal_eval(data)))
    y_vdn_promotion = y_vdn_promotion[:length]
    x_vdn_promotion = np.array(range(len(y_vdn_promotion)))

plt.figure(figsize = (10, 6))

def plot_with_std(x, y, std, label, color):
    plt.plot(x, y, label = label, color = color)
    plt.fill_between(x, y - std, y + std, color = color, alpha = 0.3)

def rolling_std(data, window = 50):
    std = []
    for i in range(len(data)):
        start = max(0, i - window)
        window_data = data[start: i + 1]
        std.append(np.std(window_data))
    return 3 * np.array(std)

plot_with_std(x_vdn_base, y_vdn_base, rolling_std(y_vdn_base), 'VDN baseline', 'blue')
plot_with_std(x_qmix_base, y_qmix_base, rolling_std(y_qmix_base), 'QMIX baseline', 'orange')
plot_with_std(x_vdn_promotion, y_vdn_promotion, rolling_std(y_vdn_promotion), 'VDN promotion', 'green')

plt.xlabel('Episode')
plt.ylabel('Monster HP')
plt.title('Monster last step hp in every episode')
plt.legend()
plt.grid()
plt.savefig('monster_last_step_hp.png', dpi = 300, bbox_inches='tight')
plt.show()