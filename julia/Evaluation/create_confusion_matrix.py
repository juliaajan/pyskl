import argparse
import os
import mmcv
import numpy as np
import matplotlib.pyplot as plt

try:
    #for remote labpc
    from mmcv import Config 
    from mmcv import load
except (ImportError, ModuleNotFoundError):
    try:
        #for windows
        from mmengine.config import Config 
        from mmengine.fileio import load
    except (ImportError, ModuleNotFoundError) as e:
        raise ModuleNotFoundError(f"Weder mmcv.load noch mmengine.fileio.load konnten importiert werden. Fehlermeldung: {e}" )

#from pyskl.core.evaluation import confusion_matrix #for lab pc only


def load_gt_labels_from_ann(anno_data):
    #load annotation file

    split_name= 'test'

    if isinstance(anno_data, dict) and 'annotations' in anno_data and 'split' in anno_data:
        split_dict = anno_data['split']
        annotations = anno_data['annotations']

        #check if provided split name is in annotation file
        if split_name not in split_dict:
            raise KeyError(f"Split '{split_name}' not found in annotation file")

        #create a set with all id's that belong to the specified split_name
        id_from_desired_split = set(split_dict[split_name])
        #only get the annotations that belong to the specified split_name
        annotations = [x for x in annotations if x['frame_dir'] in id_from_desired_split]
    else:
        raise TypeError('Unsupported annotation format in ann_file.')

    #extract labels (true classes) from annotations
    labels = np.asarray([int(x['label']) for x in annotations], dtype=np.int64)
    return labels


def load_pred_labels(pred_file):
    preds = load(pred_file)

    if not isinstance(preds, list):
        raise TypeError(f'Prediction file must contain a list, got {type(preds)}')
    if len(preds) == 0:
        raise ValueError('Prediction list is empty.')

    #check if predictions are stored as numpy array, get predicted classes
    if isinstance(preds[0], np.ndarray):
        pred_labels = np.argmax(np.stack(preds), axis=1).astype(np.int64)
        return pred_labels

    raise TypeError(
        'Unsupported prediction format. Expected list[np.ndarray] of class scores.')


def expand_to_num_classes(cm, y_pred, y_real, num_classes):
    if num_classes is None or cm.shape == (num_classes, num_classes):
        return cm

    print("EXPANDING CONFUSION MATRIX TO NUM_CLASSES")
    print("Original shape:", cm.shape)
    print("Target shape:", (num_classes, num_classes))
    #create a set with all labels that are present in the predictions and ground truth
    label_set = np.unique(np.concatenate((y_pred, y_real)))
    #create an empty matrix in the target shape num_classes x num_classes
    full_cm = np.zeros((num_classes, num_classes), dtype=cm.dtype)

    #iterate through the original confusion matrix and copy values to the correct position in the new matrix
    #geht zweimal (innen und außen) durch alle label (label_set)
    #hole dann den wert der confusion matrix an der stelle (i, j) und füge ihn an der stelle (true_label, pred_label) in der neuen matrix ein, aber nur wenn true_label und pred_label kleiner als num_classes sind, da die neue matrix nur num_classes x num_classes groß ist
    for i, true_label in enumerate(label_set):
        for j, pred_label in enumerate(label_set):
            if true_label < num_classes and pred_label < num_classes:
                full_cm[true_label, pred_label] = cm[i, j]
    return full_cm

def analyze_missing_classes(pred_labels, gt_labels, num_classes, labels_dir):

    #pred_labels and gt_labels are as many as there are test videos (here: 530), not as many as I have classes

    #get all unique class labels that were predicted and used as ground truth labels
    unique_used_gt_labels = set(np.unique(gt_labels))
    unique_used_pred_labels = set(np.unique(pred_labels))

    all_labels = load_labels(labels_dir) #load labels from labels_mapping.txt, returns e.g. "246 small"
    label_set = {int(l.split(maxsplit=1)[0]) for l in all_labels if l.strip()}

    #save a mapping of the label ids with their label names
    label_map = {}
    for l in all_labels:
        if not l.strip():
            continue
        label_id, label_name = l.split(maxsplit=1)
        label_map[int(label_id)] = label_name


    #classes from the original wlasl label list that are not in the ground truth labels
    missing_gt = sorted(label_set - unique_used_gt_labels)
    print(f"Classes in the ground truth: {len(unique_used_gt_labels)}/{num_classes}, {len(missing_gt)} are missing")
    print(f"Missing Ground truth classes: {missing_gt}")
    #also print out corresponding label names
    for label_id in missing_gt:
        label_name = label_map.get(label_id, "Unknown")
        print(f"Missing ground truth class with labels: {label_id} - {label_name}")

    #classes from the ground truth that were never predicted
    #missing_prediction = unique_used_gt_labels - unique_used_pred_labels
    missing_prediction = label_set - unique_used_pred_labels
    missing_prediction = sorted(int(x) for x in missing_prediction)
    print(f"Classes predicted: {len(unique_used_pred_labels)}/{num_classes}, {len(missing_prediction)} are missing")
    print(f"Missing Predicted classes: {missing_prediction}")

    #also print out corresponding label names
    for label_id in missing_prediction:
        label_name = label_map.get(label_id, "Unknown")
        print(f"Missing predicted class with labels: {label_id} - {label_name}")


    #check why the original confusion matrix shape is 295x295:
    all_set = set(range(num_classes))
    gt_not_pred = sorted(unique_used_gt_labels - unique_used_pred_labels)
    pred_not_gt = sorted(unique_used_pred_labels - unique_used_gt_labels)
    union_missing= sorted(all_set - (unique_used_gt_labels | unique_used_pred_labels))
    print("######")
    print("len(union):", len(unique_used_gt_labels | unique_used_pred_labels))  # muss cm.shape[0] sein
    print("missing_from_union:", union_missing) 

    #TODO: das ganze als textdatei speichern





def annotate_matrix_values(cm, mode):
    threshold = float(np.max(cm)) / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            #get the cell value from the confusion matrix
            val = cm[i, j]
            #depending on the normalization mode, format the value
            if mode in {'true', 'pred'}:
                text = f"{val:.2f}" if val > 0 else ""
            elif mode == 'all':
                if val == 0:
                    text = ""
                elif val < 1e-4:
                    text = f"{val:.1e}"
                else:
                    text = f"{val:.4f}"
            elif mode == 'raw':
                #take the raw count value and round (if needed) for non-normalized conusion matrix
                text = f"{int(round(val))}" if val > 0 else ""
            text_color = 'white' if val < threshold else 'black'
            plt.text(j, i, text, ha='center', va='center', color=text_color, fontsize=8)

#Visualize
def visualize_confusion_matrix(cm, out_dir, mode, classes_per_plot, labels_dir, show, show_values=False):
    plt.figure(figsize=(20, 20))

    im = plt.imshow(cm, interpolation='nearest', cmap='viridis')
    file_path = os.path.join(out_dir, f'cm_all_classes_{mode}.png') 
  
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks([])
    plt.yticks([])
    plt.savefig(file_path, bbox_inches='tight')
    if show:
        plt.show()
    plt.close()


    #visualise subsets of the confusion matrix for better visibility
    if classes_per_plot is not None and classes_per_plot < cm.shape[0]:
        num_classes = cm.shape[0]

        #load the labels to annotate the axes
        #TODO: Methode nutzen labels = load_pred_labels(labels_dir), geht iwie noch nicht
        if labels_dir is not None:
            with open(labels_dir, encoding='utf-8') as f:
                labels = [line.replace('\t', ' ').strip() for line in f]
        else:
            labels = None

        #create subsets of the confusion matrix and save them as separate images
        for i in range(0, num_classes, classes_per_plot):
            plt.figure(figsize=(20, 20))
            subset_cm = cm[i:i+classes_per_plot, i:i+classes_per_plot]
            im = plt.imshow(subset_cm, interpolation='nearest', cmap='viridis')
            file_path = os.path.join(out_dir, f'cm_classes_{i}_to_{i+classes_per_plot-1}_{mode}.png')
            plt.title(f'Confusion Matrix for Classes {i} to {i+classes_per_plot-1}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')

            #add the labels to y and x axes
            if labels is not None:
                plt.xticks(ticks=np.arange(classes_per_plot), labels=labels[i:i+classes_per_plot], rotation=90)
                plt.yticks(ticks=np.arange(classes_per_plot), labels=labels[i:i+classes_per_plot])
            else:
                plt.axis('off')

            #add the actual numeric values to the cells
            if show_values:
                annotate_matrix_values(subset_cm, mode)

            plt.savefig(file_path, bbox_inches='tight')
            if show:
                plt.show()
            plt.close()




#importiert vom Python Module, wieder wegnehmen um auf lab-pc auszuführen
def  confusion_matrix(y_pred, y_real, normalize=None):
    """Compute confusion matrix.

    Args:
        y_pred (list[int] | np.ndarray[int]): Prediction labels.
        y_real (list[int] | np.ndarray[int]): Ground truth labels.
        normalize (str | None): Normalizes confusion matrix over the true
            (rows), predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized. Options are
            "true", "pred", "all", None. Default: None.

    Returns:
        np.ndarray: Confusion matrix.
    """
    if normalize not in ['true', 'pred', 'all', None]:
        raise ValueError("normalize must be one of {'true', 'pred', "
                         "'all', None}")

    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)
    if not isinstance(y_pred, np.ndarray):
        raise TypeError(
            f'y_pred must be list or np.ndarray, but got {type(y_pred)}')
    if not y_pred.dtype == np.int64:
        raise TypeError(
            f'y_pred dtype must be np.int64, but got {y_pred.dtype}')

    if isinstance(y_real, list):
        y_real = np.array(y_real)
    if not isinstance(y_real, np.ndarray):
        raise TypeError(
            f'y_real must be list or np.ndarray, but got {type(y_real)}')
    if not y_real.dtype == np.int64:
        raise TypeError(
            f'y_real dtype must be np.int64, but got {y_real.dtype}')

    label_set = np.unique(np.concatenate((y_pred, y_real)))
    num_labels = len(label_set)
    max_label = label_set[-1]
    label_map = np.zeros(max_label + 1, dtype=np.int64)
    for i, label in enumerate(label_set):
        label_map[label] = i

    y_pred_mapped = label_map[y_pred]
    y_real_mapped = label_map[y_real]

    confusion_mat = np.bincount(
        num_labels * y_real_mapped + y_pred_mapped,
        minlength=num_labels**2).reshape(num_labels, num_labels)

    with np.errstate(all='ignore'):
        if normalize == 'true':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=1, keepdims=True))
        elif normalize == 'pred':
            confusion_mat = (
                confusion_mat / confusion_mat.sum(axis=0, keepdims=True))
        elif normalize == 'all':
            confusion_mat = (confusion_mat / confusion_mat.sum())
        confusion_mat = np.nan_to_num(confusion_mat)

    return confusion_mat



def load_labels(labels_dir):
    #load class labels if provided
    #returns labels prefixed with their id, e.g. "246 small"
    if labels_dir is not None:
        with open(labels_dir, encoding='utf-8') as f:
            labels = [line.replace('\t', ' ').strip() for line in f]
    else:
        labels = None
    return labels



def parse_args():
    parser = argparse.ArgumentParser(description='Create confusion matrices from *_pred.pkl outputs')
    parser.add_argument('config', help='Path to training config file')
    parser.add_argument('pred', help='Path to prediction file, e.g. best_pred.pkl')
    parser.add_argument('out_dir', help='Path where to save the confusion matrices')
    parser.add_argument('--num_classes',type=int,default=None, help='Matrix size / number of classes (e.g. 300 fpr WLASL300. If not provided, it will be taken from the config file')
    parser.add_argument('--classes_per_plot',type=int,default=None, help='Size of the subset of classes to plot in one confusion matrix. If not provided, only one cm for all classes will be generated')
    parser.add_argument('--labels_dir', type=str, default=None, help='Path to the directory containing class labels')
    parser.add_argument('--show', action='store_true', help='Whether to display the confusion matrices')
    parser.add_argument('--show_values', action='store_true', help='Annotate each cell with its numeric value')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    try:
        ann_file = cfg.data.test.ann_file
        data = load(ann_file)
    except FileNotFoundError:
        ann_file = "C:\\Users\\juja\\Desktop\\Julia\\Studium\\Uni\\Master\\Masterarbeit\\Code\\Backup-cvpc-04-03\\WLASL300\\pyskl_mediapipe_annos_2d_denormalized_NOFACE.pkl"
        data= load(ann_file)
        print("WARNUNG: anno file konnte nicht an im Config verlinkter Stelle gefunden werden. Stattdessen wird anno file von folgendem Pfad genutzt: ")
        print(ann_file)

    #get the amount of classes from the config file if not provided as argument
    if args.num_classes is not None:
        num_classes = args.num_classes
    else:
        num_classes = cfg.model.get('cls_head', {}).get('num_classes', None)

    #ground truth labels
    gt_labels = load_gt_labels_from_ann(data)
    #predicted lables from training/test output
    pred_labels = load_pred_labels(args.pred)

    if len(gt_labels) != len(pred_labels):
        raise ValueError(
            f'Length mismatch: {len(gt_labels)} GT labels vs {len(pred_labels)} predictions')

    os.makedirs(args.out_dir, exist_ok=True)


    #normalize options: None, true, pred, all
    cm_raw = confusion_matrix(pred_labels, gt_labels, normalize=None)
    cm_true = confusion_matrix(pred_labels, gt_labels, normalize='true')
    cm_pred = confusion_matrix(pred_labels, gt_labels, normalize='pred')
    cm_all = confusion_matrix(pred_labels, gt_labels, normalize='all')

    cm_raw = expand_to_num_classes(cm_raw, pred_labels, gt_labels, num_classes)
    cm_true = expand_to_num_classes(cm_true, pred_labels, gt_labels, num_classes)
    cm_pred = expand_to_num_classes(cm_pred, pred_labels, gt_labels, num_classes)
    cm_all = expand_to_num_classes(cm_all, pred_labels, gt_labels, num_classes)

    #np.save(os.path.join(args.out_dir, 'confusion_matrix_raw.npy'), cm_raw)
    np.save(os.path.join(args.out_dir, 'confusion_matrix_norm_true.npy'), cm_true)
    #np.save(os.path.join(args.out_dir, 'confusion_matrix_norm_pred.npy'), cm_pred)
    #np.save(os.path.join(args.out_dir, 'confusion_matrix_norm_all.npy'), cm_all)

    #also export as csv 
    #np.savetxt(os.path.join(args.out_dir, 'confusion_matrix_raw.csv'), cm_raw, delimiter=',', fmt='%.8f')
    np.savetxt(os.path.join(args.out_dir, 'confusion_matrix_norm_true.csv'), cm_true, delimiter=',', fmt='%.8f')
    #np.savetxt(os.path.join(args.out_dir, 'confusion_matrix_norm_pred.csv'), cm_pred, delimiter=',', fmt='%.8f')
    #np.savetxt(os.path.join(args.out_dir, 'confusion_matrix_norm_all.csv'), cm_all, delimiter=',', fmt='%.8f')

    #qualitative analysis
    #analyze_missing_classes(pred_labels, gt_labels, num_classes, args.labels_dir)

    #visualize as heatmap
    #visualize_confusion_matrix(cm_raw, args.out_dir, 'raw',args.classes_per_plot, args.labels_dir, args.show, args.show_values)
    visualize_confusion_matrix(cm_true, args.out_dir, 'true', args.classes_per_plot, args.labels_dir, args.show, args.show_values)
    #visualize_confusion_matrix(cm_pred, args.out_dir, 'pred' args.classes_per_plot, args.labels_dir, args.show, args.show_values)
    visualize_confusion_matrix(cm_all, args.out_dir, 'all', args.classes_per_plot, args.labels_dir, args.show, args.show_values)

    print('Saved confusion matrices to:', args.out_dir)
    print('Raw shape:', cm_raw.shape)
