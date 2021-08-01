from babyai.efficiency import main

total_time = int(1e6)
# for i in [1, 2, 3]:
#     # i is the random seed
#     main('BabyAI-GoToImpUnlock-v0', i, total_time, 1000000)
# 'main' will use a different seed for each of the runs in this series
main('BabyAI-UnlockRGB-v0', 1, total_time, int(2 ** 12), int(2 ** 15), step_size=2 ** 0.2, level_type='small')