import argparse
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov5l.pt', help='initial weights path')
    
    parser.add_argument('--num_train_epochs', type=int, default=300)
    # opt =  parser.parse_args(args=[])
    opt = parser.parse_args()
    return opt

def main(opt) :
    
    print('opt.weights:',opt.weights)
    print('opt.num_train_epochs:',opt.num_train_epochs)
    
    value=opt.num_train_epochs+90000
    print('value:',value)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
