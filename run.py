from PatchCore import PatchCore
from dataset import MVTecData
model = PatchCore('resnet50')
#model = model.cuda()
train_data, test_data = MVTecData('capsule',224).get_datasets()
train_dataloader = DataLoader(train_data, shuffle=True)
test_dataloader = DataLoader(test_data, shuffle=False)
# Estrazione delle features delle immagini corrette
model.fit(train_dataloader)
model.evaluate(test_dataloader)



 # Visualizzazione della curva ROC
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='Curva ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasso di falso positivo (1 - Specificità)')
    plt.ylabel('Tasso di vero positivo (Sensibilità)')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()