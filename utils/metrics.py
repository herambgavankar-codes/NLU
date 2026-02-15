def compute_metrics(y_true, y_pred):
    tp=tn=fp=fn=0
    for yt, yp in zip(y_true, y_pred):
        if yt==1 and yp==1: tp+=1
        elif yt==0 and yp==0: tn+=1
        elif yt==0 and yp==1: fp+=1
        elif yt==1 and yp==0: fn+=1

    acc = (tp+tn)/len(y_true)
    prec = tp/(tp+fp+1e-9)
    rec = tp/(tp+fn+1e-9)
    f1 = 2*prec*rec/(prec+rec+1e-9)

    return {"accuracy":acc,"precision":prec,"recall":rec,"f1":f1}
