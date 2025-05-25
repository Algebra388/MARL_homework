a = [
    ('apple', 3, 0.5),
    ('banana', 2, 0.25),
    ('orange', 5, 0.75)
]
print(*a)
# *a解包
# ('apple', 3, 0.5) ('banana', 2, 0.25) ('orange', 5, 0.75)
print(list(zip(*a)), sep = '\n')
# 把第一个，第二个分别打包，实现转置的功能