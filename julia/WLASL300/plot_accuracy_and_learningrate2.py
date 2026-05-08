import pandas as pd
import argparse
import matplotlib.pyplot as plt
import re
import numpy as np


#Adaption des plot_accuracy_learningrate.py skripts, weil ich neuen validation loss nur für die ersten 11 epochen habe,
#aber train weiterhin für 240 epochen




#as the validation loss is not logged in the .json log, we need to parse them through the .log file
def get_validation_loss(pathForVal):
    pathForVal = pathForVal.replace('.log.json', '.log')
    val_losses = []

    with open(pathForVal, 'r') as f:
        for line in f:
            #search for the correct line that loggs the validation loss
            match = re.search(r'pyskl - INFO - ValidationLoss : [0-9].[0-9]+', line)
            matchOld = re.search(r'pyskl - INFO - Loss : [0-9].[0-9]+', line)
            
            if match:
                val_loss = line.split('ValidationLoss : ')[1]
                #print(val_loss)
                val_losses.append(float(val_loss))
                
            #backup for older log files where i only logged "Loss"
            elif matchOld:
                val_loss = line.split('Loss : ')[1]
                val_losses.append(float(val_loss))
            
    print(val_losses)
    return val_losses


def plot(pathForTrain, pathForVal): 
    val_losses = get_validation_loss(pathForVal)
    print("Val Losses:")
    print(val_losses)

    #data = {"mode": "train", "epoch": 1, "iter": 20, "lr": 0.05, "memory": 5082, "data_time": 0.48623, "top1_acc": 0.0, "top5_acc": 0.0, "loss_cls": 5.7752, "loss": 5.7752, "grad_norm": 2.36381, "time": 0.84397}

    df = pd.read_json(pathForTrain, lines=True)
    df = df.dropna(axis=0, subset=['mode', 'epoch'])
    #df2 = df.groupby(["epoch", "mode" ])
    #print(df2[['mode', 'epoch', 'top1_acc', 'loss']].head())

    #df = df.groupby(["epoch", "mode" ], dropna=True)

    #agg
    agg = df.groupby(["epoch", "mode" ]).agg({
    'top1_acc': 'mean',
    'loss': 'mean' }).reset_index()
    

    print(agg[['mode', 'epoch', 'top1_acc', 'loss']].head())


    #agg.plot(kind = 'scatter', x='mode', y='loss')
    
    #plot train loss and accuracy
    print("train")
    train = df[df['mode'] == 'train'].groupby('epoch', as_index=False).agg({'top1_acc': 'mean', 'loss': 'mean'})
    print(train[['epoch', 'top1_acc', 'loss']].head())
    ax_train = train.plot(x='epoch', y='loss', label='Train Loss')
    train.plot(x='epoch', y='top1_acc', ax=ax_train, color='orange', label='Train Accuracy (top1)')

    fig_path = pathForVal.replace('.log.json', '_train_loss_accuracy.png')
    plt.savefig(fig_path)
    print(f"Saved training loss and accuracy plot to {fig_path}") 
    plt.show()


    #plot val loss and accuracy
    print("val")
    val = df[df['mode'] == 'val'].groupby('epoch', as_index=False).agg({'top1_acc': 'mean', 'loss': 'mean'})
    #add val losses from different log file
    if (len(val) != len(val_losses)):
        #raise ValueError(f"Number of validation epochs in log file ({len(val)}) does not match number of validation losses parsed from log file ({len(val_losses)}).")
        print(f"Number of validation epochs in log file ({len(val)}) does not match number of validation losses parsed from log file ({len(val_losses)}).")
        val_losses += [np.nan] * (len(val) - len(val_losses)) #fill the remaining values of val_losses with Nan to get it to the same length
    val['loss'] = val_losses
    print(val[[ 'epoch', 'top1_acc', 'loss']].head())
    ax_val = val.plot(x='epoch', y='loss', label='Val Loss')
    val.plot(x='epoch', y='top1_acc', ax=ax_val, color='orange', label='Val Accuracy (top1)')

    fig_path = pathForVal.replace('.log.json', '_val_loss_accuracy.png')
    plt.savefig(fig_path)
    print(f"Saved validation loss and accuracy plot to {fig_path}") 
    plt.show()

    #plot val and train loss together
    #train_val = df.groupby('epoch', as_index=False).agg({'loss': 'mean'})
    ax_train_loss = train.plot(x='epoch', y='loss', label='Train Loss', color="blue")
    val.plot(x='epoch', y='loss',  ax=ax_train_loss, color='orange', label='Val Loss')

    fig_path = pathForVal.replace('.log.json', '_val_train_loss.png')
    plt.savefig(fig_path)
    print(f"Saved train and validationloss plot to {fig_path}") 
    plt.show()




    



    """ for mode in agg['mode'].unique(): 
        mode_df = agg[agg['mode'] == mode] #stimmt da snoch oder muss === 'train' machen?

        print(val[[ 'epoch', 'top1_acc', 'loss']].head())
        fig, ax1 = plt.subplots()
        ax1.plot(mode_df["epoch"], mode_df["loss"], color='red', label='loss')
        ax1.set_xlabel("epoch")
        ax1.set_ylabel("loss", color='red')
        ax1.tick_params(axis='y', labelcolor='red') #???

        ax2 = ax1.twinx()
        ax2.plot(mode_df["epoch"], mode_df["top1_acc"], color='blue', label='top1_acc')
        ax2.set_ylabel("top1_acc", color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')

        ymin = min(mode_df["loss"].min(), mode_df["top1_acc"].min())
        ymax = max(mode_df["loss"].max(), mode_df["top1_acc"].max())
        ax1.set_ylim(ymin, ymax)
        ax2.set_ylim(ymin, ymax)

        plt.title(f"Mode: {mode} top1_acc and loss per epoch")
        #plt.plot(m['epoch'], m['top1_acc'], marker='o', label=mode)
        #mode_df.plot(kind = 'scatter', x='epoch', y='top1_acc')

        plt.show()
  """



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot learning rate and top1 accuracy')
    parser.add_argument('pathForTrain', type=str, help='path to json logfile, eg 20260415_170711.log.json')
    parser.add_argument('pathForVal', type=str, help='path to json logfile, eg 20260415_170711.log.json')
    args = parser.parse_args()

    plot(args.pathForTrain, args.pathForVal)