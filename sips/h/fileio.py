def init_csv(fn, header, close=True):
    '''

    '''
    f = open(fn, 'a')
    write_list(f, header)
    if close:
        f.close()
    else:
        return f


def write_list(file, list):
    if not list:
        return
    length = len(list)
    for i, elt in enumerate(list):
        file.write(str(elt))
        if i == length - 1:
            file.write('\n')
        else:
            file.write(',')
