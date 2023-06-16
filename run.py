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