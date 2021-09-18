from timeit import repeat


def find_instr(func, key):
    cnt = 0
    for txt in func.inspect_asm(func.signatures[0]).split('\n'):
        if key in txt:
            cnt += 1
            print(txt)
    if cnt == 0:
        print('Cant found instructions:', key)


def run_time(func, *args):
    # time it
    print('{:>5.0f} ms'.format(min(repeat(
        lambda: func(*args), number=5, repeat=2)) * 1000))
