import torch
from torch import nn
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch import nn
import numpy as np
import pandas as pd
import selfies as sf

def preprocess(input_file: str, output_file: str):
    # 加载CSV文件
    df = pd.read_csv(input_file)

    # 删除包含'P', 'I', 'B'的行
    df2 = df[~df['smiles'].str.contains('P|I|B|p|@|s|-|l|F|i|#|l')]
    df2 = df2.reset_index(drop=True)
    valid_indices = smiles2selfies(df2['smiles'])
    df2 = df2.iloc[valid_indices].reset_index(drop=True)

    # 存储预处理好的文件
    df2.to_csv(output_file, index=False)

# smiles是列表，selfies也是列表
def smiles2selfies(smiles):
    selfies = []
    valid_indices = []
    for i, smi in enumerate(smiles):
        if '.' not in smi:
            try:
                encoded_selfies = sf.encoder(smi)
                if encoded_selfies is not None:
                    selfies.append(encoded_selfies)
                    valid_indices.append(i)
            except sf.EncoderError:
                pass
    return selfies, valid_indices

def onehotSELFIES(selfies):
    alphabet = ['[#Branch1]', '[#Branch2]', '[#C]', '[-/Ring1]', '[-\\Ring1]', '[-\\Ring2]', '[/C]', '[/N]', '[/O]', '[/S]', '[2H]', '[3H]', '[=Branch1]', '[=Branch2]', '[=CH0]', '[=C]', '[=N]', '[=O]', '[=Ring1]', '[=Ring2]', '[=SH1]', '[=S]', '[Branch1]', '[Branch2]', '[CH0]', '[CH1]', '[CH2]', '[C]', '[NH0]', '[NH1]', '[N]', '[OH0]', '[O]', '[P]', '[Ring1]', '[Ring2]', '[S]', '[SH0]', '[\\C]', '[\\N]', '[\\O]', '[\\S]', '[nop]']

    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
    idx_to_symbol = {ch: ii for ii, ch in symbol_to_idx.items()}
    pad_to_len = 123

    # embed list of characters to list of integers
    embed_selfies = []
    for s in selfies:
        embed = sf.selfies_to_encoding(s,
                                 vocab_stoi=symbol_to_idx,
                                 pad_to_len=pad_to_len,
                                 enc_type='label')    
        embed_selfies.append(embed)

    # one hot encode
    dict_size = len(symbol_to_idx)
    seq_len = pad_to_len
    data_size = len(embed_selfies)
    sequence = embed_selfies

    # Creating a multi-dimensional array of zeros with the desired output shape
    features = np.zeros((data_size, dict_size, seq_len), dtype=np.float32)

    # Replacing the 0 at the relevant character index with a 1 to represent that character
    for i in range(data_size):
        for u in range(seq_len):
            features[i, sequence[i][u], u] = 1

    onehot_selfies = features

    return onehot_selfies, idx_to_symbol

# custom dataset
class SELFIES_Dataset(Dataset):
    def __init__(self, input_seq, target_seq, transform=None):
        self.X = input_seq
        self.y = target_seq
        self.transforms = transform
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, idx):
        
        if self.transforms:
            X = self.transforms(self.X[idx])
            y = self.transforms(self.y[idx])
            return (X, y)

        else:
            return (self.X[idx], self.y[idx])


class TextImageDataset(Dataset):
    def __init__(self, data, onehot_selfies):
        self.data = data.iloc[:, 1:1801].astype('float32').to_numpy()  # Set the data type to float32 and convert to NumPy array
        self.onehot_selfies = onehot_selfies

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # Get the image data and reshape it to a 36x50 matrix
        image_data = self.data[index].reshape(36, 50)
        image = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension

        # Get the one-hot encoded text data
        text_data = self.onehot_selfies[index]
        text = torch.tensor(text_data, dtype=torch.float32)

        return image, text

# 文本编码器
class TextEncoder(nn.Module):
    def __init__(self, params):
        super(TextEncoder, self).__init__()

        # Load Model Parameters
        self.num_characters = params['num_characters']
        self.max_seq_len = params['seq_length']
        self.num_conv_layers = params['num_conv_layers']
        self.layer1_filters = params['layer1_filters']
        self.layer2_filters = params['layer2_filters']
        self.layer3_filters = params['layer3_filters']
        self.layer4_filters = params['layer4_filters']
        self.kernel1_size = params['kernel1_size']
        self.kernel2_size = params['kernel2_size']
        self.kernel3_size = params['kernel3_size']
        self.kernel4_size = params['kernel4_size']

        # Conv1D encoding layers
        self.convl1 = nn.Conv1d(self.num_characters, self.layer1_filters, self.kernel1_size, padding=self.kernel1_size // 2)
        self.convl2 = nn.Conv1d(self.layer1_filters, self.layer2_filters, self.kernel2_size, padding=self.kernel2_size // 2)
        self.convl3 = nn.Conv1d(self.layer2_filters, self.layer3_filters, self.kernel3_size, padding=self.kernel3_size // 2)
        self.convl4 = nn.Conv1d(self.layer3_filters, self.layer4_filters, self.kernel4_size, padding=self.kernel4_size // 2)
        
        # Linear layers to connect convolutional layers to mu and logvar
        if self.num_conv_layers == 1:
            self.fc_mu = nn.Linear(self.layer1_filters * self.max_seq_len, params['latent_dimensions'])
            self.fc_logvar = nn.Linear(self.layer1_filters * self.max_seq_len, params['latent_dimensions'])
        elif self.num_conv_layers == 2:
            self.fc_mu = nn.Linear(self.layer2_filters * self.max_seq_len, params['latent_dimensions'])
            self.fc_logvar = nn.Linear(self.layer2_filters * self.max_seq_len, params['latent_dimensions'])
        elif self.num_conv_layers == 3:
            self.fc_mu = nn.Linear(self.layer3_filters * self.max_seq_len, params['latent_dimensions'])
            self.fc_logvar = nn.Linear(self.layer3_filters * self.max_seq_len, params['latent_dimensions'])
        elif self.num_conv_layers == 4:
            self.fc_mu = nn.Linear(self.layer4_filters * self.max_seq_len, params['latent_dimensions'])
            self.fc_logvar = nn.Linear(self.layer4_filters * self.max_seq_len, params['latent_dimensions'])

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.rand_like(std)
        return mu + eps * std
    
    def forward(self, x):
        if self.num_conv_layers == 1:
            x = F.relu(self.convl1(x))
        elif self.num_conv_layers == 2:
            x = F.relu(self.convl1(x))
            x = F.relu(self.convl2(x))
        elif self.num_conv_layers == 3:
            x = F.relu(self.convl1(x))
            x = F.relu(self.convl2(x))
            x = F.relu(self.convl3(x))
        elif self.num_conv_layers == 4:
            x = F.relu(self.convl1(x))
            x = F.relu(self.convl2(x))
            x = F.relu(self.convl3(x))
            x = F.relu(self.convl4(x))
            
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar
          
class VAE(nn.Module):

    def __init__(self, params):

        super(VAE, self).__init__()
        
        # Load Model Parameters
        self.num_characters = params['num_characters']
        self.max_seq_len = params['seq_length']
        self.in_dimension = params['num_characters']*params['seq_length']
        self.output_dimension = params['seq_length']
        self.num_conv_layers = params['num_conv_layers']
        self.layer1_filters = params['layer1_filters']
        self.layer2_filters = params['layer2_filters']
        self.layer3_filters = params['layer3_filters']
        self.layer4_filters = params['layer4_filters']
        self.kernel1_size = params['kernel1_size']
        self.kernel2_size = params['kernel2_size']
        self.kernel3_size = params['kernel3_size']
        self.kernel4_size = params['kernel4_size']
        self.lstm_stack_size = params['lstm_stack_size']
        self.lstm_num_neurons = params['lstm_num_neurons']
        self.latent_dimensions = params['latent_dimensions']
        self.batch_size = params['batch_size']
        
        # 添加 TextEncoder 实例
        self.text_encoder = TextEncoder(params)  # 确保传递适当的参数
        
        # Conv1D encoding layers
        self.convl1 = nn.Conv1d(self.num_characters, self.layer1_filters, self.kernel1_size, padding = self.kernel1_size//2)
        self.convl2 = nn.Conv1d(self.layer1_filters, self.layer2_filters, self.kernel2_size, padding = self.kernel2_size//2)
        self.convl3 = nn.Conv1d(self.layer2_filters, self.layer3_filters, self.kernel3_size, padding = self.kernel3_size//2)
        self.convl4 = nn.Conv1d(self.layer3_filters, self.layer4_filters, self.kernel4_size, padding = self.kernel4_size//2)

        # Linear layers to connect convolutional layers to mu and logvar
        if self.num_conv_layers == 1:
            self.fc_mu = nn.Linear(self.layer1_filters*self.max_seq_len, self.latent_dimensions)  # fc for mean of Z
            self.fc_logvar = nn.Linear(self.layer1_filters*self.max_seq_len, self.latent_dimensions)  # fc log variance of Z
        elif self.num_conv_layers == 2:
            self.fc_mu = nn.Linear(self.layer2_filters*self.max_seq_len, self.latent_dimensions)  # fc for mean of Z
            self.fc_logvar = nn.Linear(self.layer2_filters*self.max_seq_len, self.latent_dimensions)  # fc log variance of Z
        elif self.num_conv_layers == 3:
            self.fc_mu = nn.Linear(self.layer3_filters*self.max_seq_len, self.latent_dimensions)  # fc for mean of Z
            self.fc_logvar = nn.Linear(self.layer3_filters*self.max_seq_len, self.latent_dimensions)  # fc log variance of Z
        elif self.num_conv_layers == 4:
            self.fc_mu = nn.Linear(self.layer4_filters*self.max_seq_len, self.latent_dimensions)  # fc for mean of Z
            self.fc_logvar = nn.Linear(self.layer4_filters*self.max_seq_len, self.latent_dimensions)  # fc log variance of Z
            

        # LSTM decoding layers
        self.decode_RNN = nn.LSTM(
            input_size = self.latent_dimensions,
            hidden_size = self.lstm_num_neurons,
            num_layers = self.lstm_stack_size,
            batch_first = True,
            bidirectional = True)

        self.decode_FC = nn.Sequential(
            nn.Linear(2*self.lstm_num_neurons, self.output_dimension),
        )
        
        self.prob = nn.LogSoftmax(dim=1)
        
    @staticmethod
    
    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden =  weight.new_zeros(self.lstm_stack_size, batch_size, self.lstm_num_neurons).zero().to(device)
        return hidden                                
                                
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.rand_like(std)
        return mu + eps*std
    
    def encoder(self, x):
        # 使用 TextEncoder 替换现有的编码操作
        z, mu, logvar = self.text_encoder(x)
        
        return z, mu, logvar
    
    def decoder(self, z):
        rz = z.unsqueeze(1).repeat(1, self.num_characters, 1)
        l1, h = self.decode_RNN(rz)
        decoded = self.decode_FC(l1)
        x_hat = decoded
        
        return x_hat

    def forward(self, x):
        # Get results of encoder network
        x = x.squeeze(dim=1)
        z, mu, logvar = self.encoder(x)

        # Get results of decoder network
        x_hat = self.decoder(z)

                 
        return x_hat, z, mu, logvar

# VAE loss function #
def loss_function(recon_x, x, mu, logvar, KLD_alpha):   
    #BCE = F.binary_cross_entropy(recon_x, x.squeeze(dim=1), reduction='sum')
    
    inp = recon_x
    target = torch.argmax(x, dim=1)
    criterion = torch.nn.CrossEntropyLoss()
    BCE = criterion(inp, target)
    KLD = -0.5 * torch.mean(1. + logvar - mu.pow(2) - logvar.exp())
    
    #return BCE + KLD_alpha*KLD
    return BCE, KLD_alpha, KLD
    
    
# VAE training loop
def train(model, train_loader, optimizer, device, epoch, KLD_alpha):
    LOG_INTERVAL = 100
    model.train()
    train_loss = 0
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        data = data.to(device)   
        recon_data, z, mu, logvar = model(data)
        BCE, KLD_alpha, KLD = loss_function(recon_data, data.squeeze(dim=1), mu, logvar, KLD_alpha)
        loss = BCE + KLD_alpha*KLD
        loss.backward()
        cur_loss = loss.item()
        train_loss += cur_loss
        optimizer.step()
    print('====> Epoch: {} Average loss: {:.5f}'.format(epoch, train_loss / len(train_loader.dataset)))
    
    return train_loss / len(train_loader.dataset), BCE, KLD_alpha, KLD

# VAE testing loop
def test(model, test_loader, optimizer, device, epoch, KLD_alpha):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            recon_data, z, mu, logvar = model(data)
            BCE, KLD_alpha, KLD = loss_function(recon_data, data.squeeze(dim=1), mu, logvar, KLD_alpha)
            cur_loss = BCE + KLD_alpha*KLD
            test_loss += cur_loss

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.5f}'.format(test_loss))

    return test_loss

# 多级残差连接图像编码器
class ModifiedResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, dilation=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, dilation=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, dilation=5)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu3 = nn.ReLU(inplace=False)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, dilation=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU(inplace=False)
        self.conv5 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, dilation=2)
        self.bn5 = nn.BatchNorm2d(16)
        self.relu5 = nn.ReLU(inplace=False)
        self.conv6 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, dilation=2)
        self.bn6 = nn.BatchNorm2d(1)
        self.relu6 = nn.ReLU(inplace=False)
        self.conv7 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=13)
        self.bn7 = nn.BatchNorm2d(32)
        self.relu7 = nn.ReLU(inplace=False)
        self.conv8 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=21)
        self.bn8 = nn.BatchNorm2d(16)
        self.relu8 = nn.ReLU(inplace=False)
        self.conv9 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=27)
        self.bn9 = nn.BatchNorm2d(1)
        self.relu9 = nn.ReLU(inplace=False)
        
        self.fc = nn.Linear(36, 256)

    def forward(self, x):
        # x (1,32,32) -> out1 (16,30,30)
        out1 = self.relu1(self.bn1(self.conv1(x)))
        # out1 (16,30,30) -> out2 (32,26,26)
        out2 = self.relu2(self.bn2(self.conv2(out1)))
        # out2 (32,26,26) -> out3 (64,16,16)
        out3 = self.relu3(self.bn3(self.conv3(out2)))
        # out3 (32,26,26) -> out4 (64,16,16)
        out4 = self.relu4(self.bn4(self.conv4(out3)))

        out7 = self.relu7(self.bn7(self.conv7(out2)))
        out7_4 = out7 + out4
        
        # out4 (64,16,16) -> out5 (32,14,14)
        out5 = self.relu5(self.bn5(self.conv5(out7_4)))
        # out8 (16,14,14)
        out8 = self.relu8(self.bn8(self.conv8(out1)))
        out8_5 = out8 + out5
        
        # out5 (16,10,10) -> out6 (1,6,6)
        out6 = self.relu6(self.bn6(self.conv6(out8_5)))
        
        out9 = self.relu9(self.bn9(self.conv9(x)))
        # out9_6 (1,6,6)
        out9_6 = out9 + out6
        
        x = torch.flatten(out9_6, 1)
        # x (256,)
        x = self.fc(x)
        return x

class CLIP(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.image_encoder = ModifiedResNet()
        self.text_encoder = text_encoder   # 请确保已设置适当的参数
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.upsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=True)  # 放在 CLIP 的初始化里

    def encode_image(self, image):
        # image (1,36,50) -> (1,32,32) -> (32,32) -> (256,)
        image = self.upsample(image)  # 放在 encode_image 函数中
        image = image.squeeze(0)
        image_features = self.image_encoder(image)
        return image_features

    def encode_text(self, text):
        text_features = self.text_encoder(text)[0]  # Extract the encoded text features
        return text_features


    def forward(self, image, text):
        image_features = self.encode_image(image)
        text = text.squeeze(dim=1)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text

            
def clip_train(model, dataloader, device, num_epochs):
    model.train()  # 将模型设置为训练模式

    # 优化器与学习率调整
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        print(f"Epoch: {epoch + 1}/{num_epochs}")

        epoch_loss = 0
        epoch_acc = 0
        num_batches = 0

        for idx, (image, text) in enumerate(dataloader):
            image, text = image.to(device), text.to(device)

            ims = F.normalize(model.encode_image(image), dim=1)
            txt = F.normalize(model.encode_text(text), dim=1)
            image_logits = ims @ txt.t() * model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()

            optimizer.zero_grad()

            # 计算图像损失
            ims = F.normalize(model.encode_image(image), dim=1)
            image_logits = ims @ txt.t() * model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            loss.backward(retain_graph=True)

            # 计算文本损失
            txt = F.normalize(model.encode_text(text), dim=1)
            image_logits = ims @ txt.t() * model.logit_scale.exp()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth))/2
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            model.logit_scale.data.clamp_(-np.log(100), np.log(100))

            if idx % 500 == 0:
                print(f"Epoch: {epoch}, Step: {idx}, Loss: {loss.item()}, Acc: {(acc_i + acc_t) / 2 / len(image)}")

            epoch_loss += loss.item()
            epoch_acc += ((acc_i + acc_t) / 2 / len(image)).item()

            num_batches += 1

        # 计算并存储每个epoch的平均损失和平均精度
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        print(f"Epoch: {epoch + 1}, Avg Loss: {avg_loss}, Avg Acc: {avg_acc}")

    return losses, accuracies

# 在验证集上评估模型
def clip_evaluate(model, dataloader, device):
    model.eval()  # 将模型设置为评估模式
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

    with torch.no_grad():
        for idx, (image, text) in enumerate(dataloader):
            image, text = image.to(device), text.to(device)

            ims = F.normalize(model.encode_image(image), dim=1)
            txt = F.normalize(model.encode_text(text), dim=1)

            image_logits = ims @ txt.t() * model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()

            epoch_loss += loss.item()
            epoch_acc += ((acc_i + acc_t) / 2 / len(image)).item()
            num_batches += 1

    avg_loss = epoch_loss / num_batches
    avg_acc = epoch_acc / num_batches

    return avg_loss, avg_acc