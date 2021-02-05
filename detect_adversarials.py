print('Load modules...')
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score
import argparse

#processing the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--attack", default='fgsm', help="the attack method which created the adversarial examples you want to use. Either fgsm, bim, pgd, df or cw")
parser.add_argument("--detector", default='InputMFS', help="the detector youz want to use, out of InputMFS, InputPFS, LayerMFS, LayerPFS, LID, Mahalanobis")
parser.add_argument("--net", default='cif10', help="the network used for the attack, either cif10 or cif100")
parser.add_argument("--mode", default='test', help="choose test or validation case")
args = parser.parse_args()
#choose attack
attack_method = args.attack
detector = args.detector
mode = args.mode
net = args.net
scale = True
#load characteristics
print('Loading characteristics...')
characteristics = np.load('./data/characteristics/'+net+'_'+attack_method+'_'+detector+'.npy',allow_pickle=True)
characteristics_adv = np.load('./data/characteristics/'+net+'_'+attack_method+'_'+detector+'_adv.npy',allow_pickle=True)

shape = np.shape(characteristics)
k = shape[0]


adv_X_train_val, adv_X_test, adv_y_train_val, adv_y_test = train_test_split(characteristics_adv, np.ones(k), test_size=0.2, random_state=42)
b_X_train_val, b_X_test, b_y_train_val, b_y_test = train_test_split(characteristics, np.zeros(k), test_size=0.2, random_state=42)
adv_X_train, adv_X_val, adv_y_train, adv_y_val = train_test_split(adv_X_train_val, adv_y_train_val, test_size=0.2, random_state=42)
b_X_train, b_X_val, b_y_train, b_y_val = train_test_split(b_X_train_val, b_y_train_val, test_size=0.2, random_state=42)

X_train = np.concatenate(( b_X_train,adv_X_train))
y_train = np.concatenate(( b_y_train,adv_y_train))

if mode == 'test':
    X_test = np.concatenate(( b_X_test, adv_X_test))
    y_test = np.concatenate(( b_y_test,adv_y_test))
elif mode == 'validation':
    X_test = np.concatenate(( b_X_val, adv_X_val))
    y_test = np.concatenate(( b_y_val,adv_y_val))
else:
    print('Not a valid mode')

#train classifier
print('Training classifier...')

#special case
if (detector == 'LayerMFS'or detector =='LayerPFS') and net == 'cif100' and (attack_method=='cw' or attack_method=='df'):
    from cuml.svm import SVC
    scaler  = MinMaxScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test  = scaler.transform(X_test)
    if detector == 'LayerMFS':
        gamma = 0.1
        if attack_method == 'cw':
            C=1
        else:
            C=10
    else:
        C=10
        gamma = 0.01
    clf = SVC(probability=True, C=C, gamma=gamma)
else:
    clf = LogisticRegression() #normal case
    
clf.fit(X_train,y_train)

#save classifier
filename = './data/detectors/LR_'+attack_method+'_'+detector+'_'+mode+'_'+net+'.sav'
pickle.dump(clf, open(filename, 'wb'))

print('Evaluating classifier...')
prediction = clf.predict(X_test)
prediction_pr = clf.predict_proba(X_test)[:, 1]

benign_rate = 0
benign_guesses = 0
ad_guesses = 0
ad_rate = 0
for i in range(len(prediction)):
    if prediction[i] == 0:
        benign_guesses +=1
        if y_test[i]==0:
            benign_rate +=1
    else:
        ad_guesses +=1
        if y_test[i]==1:
            ad_rate +=1

acc = (benign_rate+ad_rate)/len(prediction)        
TP = 2*ad_rate/len(prediction)
TN = 2*benign_rate/len(prediction)
precision = ad_rate/ad_guesses
print('True positive rate/adversarial detetcion rate/recall/sensitivity is ', round(100*TP,2))
print('True negative rate/normal detetcion rate/selectivity is ', round(100*TN,2))
print('Precision',round(100*precision,1))
print('The accuracy is',round(100*acc,2))
print('The AUC score is', round(100*roc_auc_score(y_test,prediction_pr),2))
