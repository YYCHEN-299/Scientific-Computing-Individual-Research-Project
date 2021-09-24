def find_instr(func, key):
    """
    Print instructions if contain key word.

    Parameters
    ----------
    func : Numba function
        Numba function for find instructions

    key : str
        Key instruction word

    Returns
    -------
    Nothing
    """

    cnt = 0
    for txt in func.inspect_asm(func.signatures[0]).split('\n'):
        if key in txt:
            cnt += 1
            print(txt)
    if cnt == 0:
        print('Cant found instructions:', key)
