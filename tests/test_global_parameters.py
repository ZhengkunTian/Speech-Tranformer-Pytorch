def train():
    global global_step
    for i in range(5):
        global_step += 1
        print(global_step)

def main():
    global global_step
    global_step = 0

    for i in range(10):
        train()
if __name__ == '__main__':
    main()