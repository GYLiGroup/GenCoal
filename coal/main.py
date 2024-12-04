import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
import pandas as pd
import selfies as sf
from typing import List, Dict, Tuple, Optional, Callable
from coal import ClipIRMol

def preprocess(input_file: str, output_file: str):
    """
    Preprocesses a CSV file by cleaning up SMILES strings and saving the filtered data to an output CSV file.

    Parameters
    ----------
    input_file : str
        The path to the input CSV file containing the SMILES data.

    output_file : str
        The path where the preprocessed CSV file will be saved.

    Example
    -------
    >>> preprocess('input_data.csv', 'cleaned_data.csv')
    """
    # Step 1: Load the CSV file into a DataFrame
    df = pd.read_csv(input_file)

    # Step 2: Filter out rows where the SMILES string contains undesired characters
    # The regex pattern removes any SMILES string containing 'P', 'I', 'B' and other specific characters.
    df2 = df[~df['smiles'].str.contains('P|I|B|p|@|s|-|l|F|i|#|l')]

    # Reset the index of the DataFrame after filtering
    df2 = df2.reset_index(drop=True)

    # Step 3: Convert SMILES strings to SELFIES and check for valid entries
    # The function `smiles2selfies` is assumed to return a list of valid indices.
    valid_indices = smiles2selfies(df2['smiles'])

    # Filter the DataFrame using the valid indices
    df2 = df2.iloc[valid_indices].reset_index(drop=True)

    # Step 4: Save the cleaned DataFrame to a new CSV file
    df2.to_csv(output_file, index=False)

def smiles2selfies(smiles):
    """
    Convert a list of SMILES strings into SELFIES strings and return the valid SELFIES along with their indices.

    Parameters
    ----------
    smiles : list of str
        A list of SMILES strings.

    Returns
    -------
    selfies : list of str
        A list of valid SELFIES strings corresponding to the input SMILES.
    
    valid_indices : list of int
        A list of indices of the valid SMILES strings that were successfully converted to SELFIES.
    """
    selfies = []  # To store valid SELFIES strings
    valid_indices = []  # To store indices of the valid SMILES
    
    for i, smi in enumerate(smiles):
        # Skip SMILES that contain a period ('.') character (e.g., mixture of molecules)
        if '.' not in smi:
            try:
                # Try to encode the SMILES into a SELFIES string
                encoded_selfies = sf.encoder(smi)
                
                # If the encoding is successful, append the SELFIES and the index
                if encoded_selfies is not None:
                    selfies.append(encoded_selfies)
                    valid_indices.append(i)
            except sf.EncoderError:
                # If an error occurs during encoding, just skip this SMILES string
                pass
    
    return selfies, valid_indices

def onehotSELFIES(selfies):
    """
    Converts a list of SELFIES strings into one-hot encoded representations.

    Parameters
    ----------
    selfies : list of str
        A list of SELFIES strings to be one-hot encoded.

    Returns
    -------
    onehot_selfies : numpy.ndarray
        A 3D numpy array of one-hot encoded SELFIES strings with shape 
        (data_size, dict_size, seq_len), where:
        - data_size is the number of SELFIES strings.
        - dict_size is the size of the alphabet.
        - seq_len is the maximum length of the encoded SELFIES strings.
        
    idx_to_symbol : dict
        A dictionary mapping the index back to the symbol for each character.
    """
    # Define the alphabet (all possible symbols for SELFIES)
    alphabet = ['[#Branch1]', '[#Branch2]', '[#C]', '[-/Ring1]', '[-\\Ring1]', '[-\\Ring2]', 
                '[/C]', '[/N]', '[/O]', '[/S]', '[2H]', '[3H]', '[=Branch1]', '[=Branch2]', 
                '[=CH0]', '[=C]', '[=N]', '[=O]', '[=Ring1]', '[=Ring2]', '[=SH1]', '[=S]', 
                '[Branch1]', '[Branch2]', '[CH0]', '[CH1]', '[CH2]', '[C]', '[NH0]', '[NH1]', 
                '[N]', '[OH0]', '[O]', '[P]', '[Ring1]', '[Ring2]', '[S]', '[SH0]', '[\\C]', 
                '[\\N]', '[\\O]', '[\\S]', '[nop]']

    # Create mapping from symbol to index and vice versa
    symbol_to_idx = {s: i for i, s in enumerate(alphabet)}
    idx_to_symbol = {ch: ii for ii, ch in symbol_to_idx.items()}
    
    # Set the padding length to 123
    pad_to_len = 123

    # Convert SELFIES strings to their integer encodings
    embed_selfies = []
    for s in selfies:
        embed = sf.selfies_to_encoding(s, vocab_stoi=symbol_to_idx, pad_to_len=pad_to_len, enc_type='label')
        embed_selfies.append(embed)

    # Prepare the one-hot encoding
    dict_size = len(symbol_to_idx)  # Size of the alphabet
    seq_len = pad_to_len            # Padding length
    data_size = len(embed_selfies)  # Number of SELFIES strings
    sequence = embed_selfies        # Encoded SELFIES strings

    # Create a numpy array of zeros to hold the one-hot encoded features
    features = np.zeros((data_size, dict_size, seq_len), dtype=np.float32)

    # One-hot encode: set the relevant position to 1 for each character in the SELFIES string
    for i in range(data_size):
        for u in range(seq_len):
            features[i, sequence[i][u], u] = 1

    # The result is a one-hot encoded numpy array
    onehot_selfies = features

    return onehot_selfies, idx_to_symbol

class SELFIES_Dataset(Dataset):
    """
    A custom dataset class for handling pairs of input and target sequences (SELFIES) for machine learning tasks.
    
    Parameters
    ----------
    input_seq : list of str
        The list of input SELFIES sequences (usually the features or inputs to the model).
        
    target_seq : list of str
        The list of target SELFIES sequences (usually the labels or outputs from the model).
        
    transform : Callable, optional
        A transformation function to apply on each input and target sequence. By default, no transformation is applied.
    """
    
    def __init__(self, input_seq: List[str], target_seq: List[str], transform: Optional[Callable] = None) -> None:

        """
        Initializes the dataset with input and target sequences and an optional transformation function.
        
        Parameters
        ----------
        input_seq : list of str
            The list of input SELFIES sequences.
            
        target_seq : list of str
            The list of target SELFIES sequences.
        
        transform : Callable, optional
            A transformation function to apply on each input and target sequence.
        """
        self.X = input_seq
        self.y = target_seq
        self.transforms = transform

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        
        Returns
        -------
        int
            The length of the dataset (number of input-output pairs).
        """
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple:
        """
        Returns a single sample from the dataset, with optional transformations applied.
        
        Parameters
        ----------
        idx : int
            The index of the sample to retrieve.
        
        Returns
        -------
        Tuple
            A tuple containing the input sequence and the corresponding target sequence.
            If transforms are applied, the sequences are transformed before returning.
        """
        if self.transforms:
            X = self.transforms(self.X[idx])
            y = self.transforms(self.y[idx])
            return X, y
        else:
            return self.X[idx], self.y[idx]

class TextImageDataset(Dataset):
    """
    A custom dataset for handling pairs of image data and one-hot encoded text data (SELFIES).
    
    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing image data, where the columns (except the first one) represent pixel values.
        
    onehot_selfies : np.ndarray
        A NumPy array containing the one-hot encoded SELFIES sequences for each sample.
    """
    
    def __init__(self, data: pd.DataFrame, onehot_selfies: np.ndarray) -> None:
        """
        Initializes the dataset with image data and one-hot encoded SELFIES sequences.
        
        Parameters
        ----------
        data : pd.DataFrame
            The DataFrame containing image data (columns from the second column onward represent pixel values).
            
        onehot_selfies : np.ndarray
            The array of one-hot encoded SELFIES sequences corresponding to each image.
        """
        # Convert the data to NumPy array of type float32 (exclude the first column which might be labels or IDs)
        self.data = data.iloc[:, 1:1801].astype('float32').to_numpy()
        self.onehot_selfies = onehot_selfies

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        
        Returns
        -------
        int
            The length of the dataset (number of samples).
        """
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns a single sample from the dataset, consisting of image data and corresponding one-hot text data.
        
        Parameters
        ----------
        index : int
            The index of the sample to retrieve.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            A tuple containing the image data (torch tensor) and the corresponding one-hot encoded text data (torch tensor).
        """
        # Get the image data and reshape it to a 36x50 matrix
        image_data = self.data[index].reshape(36, 50)
        image = torch.tensor(image_data, dtype=torch.float32).unsqueeze(0)  # Add a channel dimension

        # Get the one-hot encoded text data
        text_data = self.onehot_selfies[index]
        text = torch.tensor(text_data, dtype=torch.float32)

        return image, text

class TextEncoder(nn.Module):
    """
    A convolutional encoder for text sequences, typically used in variational autoencoders (VAEs).
    
    Parameters
    ----------
    params : dict
        A dictionary containing model parameters, such as the number of characters (num_characters),
        sequence length (seq_length), number of convolutional layers (num_conv_layers), number of filters
        for each layer, kernel sizes, and latent space dimensions.
    """
    
    def __init__(self, params: Dict[str, int]) -> None:
        """
        Initializes the TextEncoder with the provided parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary containing hyperparameters like number of characters, sequence length,
            number of convolutional layers, filter sizes, and latent dimensions.
        """
        super(TextEncoder, self).__init__()

        # Load model parameters
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
        
        # Define convolutional layers (Conv1D)
        self.convl1 = nn.Conv1d(self.num_characters, self.layer1_filters, self.kernel1_size, padding=self.kernel1_size // 2)
        self.convl2 = nn.Conv1d(self.layer1_filters, self.layer2_filters, self.kernel2_size, padding=self.kernel2_size // 2)
        self.convl3 = nn.Conv1d(self.layer2_filters, self.layer3_filters, self.kernel3_size, padding=self.kernel3_size // 2)
        self.convl4 = nn.Conv1d(self.layer3_filters, self.layer4_filters, self.kernel4_size, padding=self.kernel4_size // 2)
        
        # Define fully connected layers to output mean (mu) and log variance (logvar)
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

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from a normal distribution N(mu, sigma^2).
        
        Parameters
        ----------
        mu : torch.Tensor
            The mean of the distribution.
        
        logvar : torch.Tensor
            The log variance of the distribution.
        
        Returns
        -------
        torch.Tensor
            A sample from the distribution using the reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)  # Calculate the standard deviation (sigma)
        eps = torch.rand_like(std)  # Generate random noise with the same shape as std
        return mu + eps * std  # Return the sample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder network. Applies the specified number of convolutional layers 
        followed by fully connected layers to output mu, logvar, and the latent representation z.
        
        Parameters
        ----------
        x : torch.Tensor
            The input tensor (one-hot encoded text).
        
        Returns
        -------
        torch.Tensor
            A tuple (z, mu, logvar) where z is the latent representation, and mu, logvar are used in the VAE.
        """
        # Pass through the convolutional layers based on the specified number of layers
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
        
        # Flatten the output from the conv layers
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        
        # Get the mean and log variance for the latent space
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        
        # Reparameterization trick
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar
          
class VAE(nn.Module):
    def __init__(self, params: Dict[str, int]) -> None:
        super(VAE, self).__init__()
        
        # Load model parameters
        self.num_characters = params['num_characters']
        self.max_seq_len = params['seq_length']
        self.in_dimension = self.num_characters * self.max_seq_len
        self.output_dimension = self.max_seq_len
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
        
        # Initialize TextEncoder
        self.text_encoder = TextEncoder(params)
        
        # Conv1D encoding layers
        self.convl1 = nn.Conv1d(self.num_characters, self.layer1_filters, self.kernel1_size, padding=self.kernel1_size // 2)
        self.convl2 = nn.Conv1d(self.layer1_filters, self.layer2_filters, self.kernel2_size, padding=self.kernel2_size // 2)
        self.convl3 = nn.Conv1d(self.layer2_filters, self.layer3_filters, self.kernel3_size, padding=self.kernel3_size // 2)
        self.convl4 = nn.Conv1d(self.layer3_filters, self.layer4_filters, self.kernel4_size, padding=self.kernel4_size // 2)

        # Fully connected layers for `mu` and `logvar`
        if self.num_conv_layers == 1:
            self.fc_mu = nn.Linear(self.layer1_filters * self.max_seq_len, self.latent_dimensions)
            self.fc_logvar = nn.Linear(self.layer1_filters * self.max_seq_len, self.latent_dimensions)
        elif self.num_conv_layers == 2:
            self.fc_mu = nn.Linear(self.layer2_filters * self.max_seq_len, self.latent_dimensions)
            self.fc_logvar = nn.Linear(self.layer2_filters * self.max_seq_len, self.latent_dimensions)
        elif self.num_conv_layers == 3:
            self.fc_mu = nn.Linear(self.layer3_filters * self.max_seq_len, self.latent_dimensions)
            self.fc_logvar = nn.Linear(self.layer3_filters * self.max_seq_len, self.latent_dimensions)
        elif self.num_conv_layers == 4:
            self.fc_mu = nn.Linear(self.layer4_filters * self.max_seq_len, self.latent_dimensions)
            self.fc_logvar = nn.Linear(self.layer4_filters * self.max_seq_len, self.latent_dimensions)

        # LSTM decoding layers
        self.decode_RNN = nn.LSTM(
            input_size=self.latent_dimensions,
            hidden_size=self.lstm_num_neurons,
            num_layers=self.lstm_stack_size,
            batch_first=True,
            bidirectional=True)

        self.decode_FC = nn.Sequential(
            nn.Linear(2 * self.lstm_num_neurons, self.output_dimension),
        )

        self.prob = nn.LogSoftmax(dim=1)

    def init_hidden(self, batch_size: int) -> torch.Tensor:
        """
        Initialize the hidden state for the LSTM.
        
        Parameters
        ----------
        batch_size : int
            The size of the batch.
        
        Returns
        -------
        torch.Tensor
            The initial hidden state for the LSTM.
        """
        weight = next(self.parameters()).data
        hidden = weight.new_zeros(self.lstm_stack_size * 2, batch_size, self.lstm_num_neurons).to(device)
        return hidden                                

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from a normal distribution N(mu, sigma^2).
        
        Parameters
        ----------
        mu : torch.Tensor
            The mean of the distribution.
        
        logvar : torch.Tensor
            The log variance of the distribution.
        
        Returns
        -------
        torch.Tensor
            A sample from the distribution using the reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)  # Calculate the standard deviation (sigma)
        eps = torch.rand_like(std)  # Generate random noise with the same shape as std
        return mu + eps * std  # Return the sample

    def encoder(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The encoder network that processes the input sequence and returns latent variable `z`, `mu`, and `logvar`.
        
        Parameters
        ----------
        x : torch.Tensor
            The input sequence.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing the latent variable `z`, the mean `mu`, and log variance `logvar`.
        """
        z, mu, logvar = self.text_encoder(x)
        return z, mu, logvar

    def decoder(self, z: torch.Tensor) -> torch.Tensor:
        """
        The decoder network that takes the latent variable `z` and generates a reconstructed sequence.
        
        Parameters
        ----------
        z : torch.Tensor
            The latent variable.
        
        Returns
        -------
        torch.Tensor
            The reconstructed sequence `x_hat`.
        """
        rz = z.unsqueeze(1).repeat(1, self.num_characters, 1)
        l1, _ = self.decode_RNN(rz)
        decoded = self.decode_FC(l1)
        x_hat = decoded
        return x_hat

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        The forward pass of the VAE model. Encodes the input and decodes it back into a reconstructed sequence.
        
        Parameters
        ----------
        x : torch.Tensor
            The input sequence.
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            The reconstructed sequence `x_hat`, latent variable `z`, mean `mu`, and log variance `logvar`.
        """
        x = x.squeeze(dim=1)
        z, mu, logvar = self.encoder(x)  # Get the latent variables from the encoder
        x_hat = self.decoder(z)  # Reconstruct the sequence using the decoder
        return x_hat, z, mu, logvar

def loss_function(recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor, KLD_alpha: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    VAE Loss function: A combination of Reconstruction Loss and KL Divergence.
    
    Parameters
    ----------
    recon_x : torch.Tensor
        The reconstructed input from the decoder (predicted).
    x : torch.Tensor
        The true input (target).
    mu : torch.Tensor
        The mean of the latent variable distribution.
    logvar : torch.Tensor
        The log variance of the latent variable distribution.
    KLD_alpha : float
        The weight for the KL Divergence loss.

    Returns
    -------
    BCE : torch.Tensor
        The Binary Cross-Entropy (or other reconstruction loss) term.
    KLD_alpha : float
        The weight for the KL Divergence loss (for reference).
    KLD : torch.Tensor
        The Kullback-Leibler Divergence loss.
    """
    
    # Reconstruction Loss (BCE for classification tasks)
    # Assuming recon_x contains logits and x is one-hot encoded
    target = torch.argmax(x, dim=1)  # Get the class indices from the one-hot encoded x
    criterion = torch.nn.CrossEntropyLoss()
    BCE = criterion(recon_x, target)  # Using CrossEntropyLoss for classification tasks
    
    # KL Divergence (regularization term)
    KLD = -0.5 * torch.mean(1. + logvar - mu.pow(2) - logvar.exp())
    
    # Return the loss components
    return BCE, KLD_alpha, KLD
  
def train(model: nn.Module, 
          train_loader: DataLoader, 
          optimizer: optim.Optimizer, 
          device: torch.device, 
          epoch: int, 
          KLD_alpha: float) -> tuple:
    """
    Train the Variational Autoencoder (VAE) model for one epoch on the given training data.

    Args:
        model (nn.Module): The VAE model to train.
        train_loader (DataLoader): The DataLoader instance that provides the training data in batches.
        optimizer (optim.Optimizer): The optimizer used to update the model's weights (e.g., Adam).
        device (torch.device): The device (CPU or GPU) on which the model and data will be placed.
        epoch (int): The current epoch number. Used for logging.
        KLD_alpha (float): A scaling factor for the Kullback-Leibler Divergence (KLD) term in the loss function.

    Returns
        tuple: A tuple containing:
            - avg_train_loss (float): The average loss for the epoch.
            - avg_BCE (float): The average Binary Cross-Entropy (BCE) loss across the epoch.
            - avg_KLD (float): The average Kullback-Leibler Divergence (KLD) loss across the epoch.
    
    This function performs the training loop for a single epoch:
        1. Zero the gradients for the optimizer.
        2. Pass data through the VAE model and calculate reconstruction loss (BCE) and KL divergence (KLD).
        3. Compute the total loss (BCE + KLD).
        4. Perform backpropagation to update the model weights.
        5. Track the total loss for logging and return average loss values after the epoch.
    """
    LOG_INTERVAL = 100  # Log every 100 steps for progress monitoring
    model.train()  # Set the model to training mode
    train_loss = 0  # Initialize total loss accumulator
    BCE_total = 0  # Initialize total BCE loss accumulator
    KLD_total = 0  # Initialize total KLD loss accumulator
    
    # Loop through batches in the training dataset
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()  # Clear previous gradients
        data = data.to(device)  # Move data to the appropriate device
        
        # Forward pass: Get the model's output (reconstructed data, z, mu, logvar)
        recon_data, z, mu, logvar = model(data)
        
        # Calculate the loss components (BCE and KLD)
        BCE, KLD_alpha, KLD = loss_function(recon_data, data.squeeze(dim=1), mu, logvar, KLD_alpha)
        
        # Total loss = reconstruction loss (BCE) + KL divergence (KLD)
        loss = BCE + KLD_alpha * KLD
        
        # Backward pass: Compute gradients for the model's parameters
        loss.backward()
        
        # Step the optimizer to update the model's parameters
        optimizer.step()
        
        # Accumulate the losses for logging
        train_loss += loss.item()
        BCE_total += BCE.item()
        KLD_total += KLD.item()
        
        # Log every LOG_INTERVAL batches
        if batch_idx % LOG_INTERVAL == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] '
                  f'Loss: {loss.item():.6f} BCE: {BCE.item():.6f} KLD: {KLD.item():.6f}')
    
    # Compute average loss values for the epoch
    avg_train_loss = train_loss / len(train_loader.dataset)
    avg_BCE = BCE_total / len(train_loader.dataset)
    avg_KLD = KLD_total / len(train_loader.dataset)
    
    # Log average training loss for the epoch
    print(f'====> Epoch: {epoch} Average loss: {avg_train_loss:.5f} '
          f'Average BCE: {avg_BCE:.5f} Average KLD: {avg_KLD:.5f}')
    
    # Return average losses for analysis or logging
    return avg_train_loss, avg_BCE, avg_KLD

def test(model: nn.Module, 
         test_loader: DataLoader, 
         optimizer: optim.Optimizer, 
         device: torch.device, 
         epoch: int, 
         KLD_alpha: float) -> float:
    """
    Evaluate the VAE model on the test dataset after training.

    Args:
        model (nn.Module): The VAE model to evaluate.
        test_loader (DataLoader): The DataLoader instance that provides the test data in batches.
        optimizer (optim.Optimizer): The optimizer used for training, though not needed for testing.
        device (torch.device): The device (CPU or GPU) on which the model and data will be placed.
        epoch (int): The current epoch number. Used for logging.
        KLD_alpha (float): A scaling factor for the Kullback-Leibler Divergence (KLD) term in the loss function, which should be the same as used in training.

    Returns
        float: The average test loss for the entire test set.

    This function performs the evaluation (testing) loop:
        1. Sets the model to evaluation mode using `model.eval()`.
        2. Iterates over the batches in the test data using `test_loader`.
        3. For each batch, it passes the input data through the model, computes the reconstruction loss (BCE) and KL divergence (KLD).
        4. Accumulates the total loss across all batches.
        5. Computes and logs the average test loss for the entire test set.
    """
    model.eval()  # Set the model to evaluation mode
    test_loss = 0  # Initialize total loss accumulator
    
    with torch.no_grad():  # Disable gradient computation to save memory and computation
        # Loop through each batch in the test data
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)  # Move data to the appropriate device
            
            # Forward pass: Get the model's output (reconstructed data, z, mu, logvar)
            recon_data, z, mu, logvar = model(data)
            
            # Calculate the loss components (BCE and KLD)
            BCE, KLD_alpha, KLD = loss_function(recon_data, data.squeeze(dim=1), mu, logvar, KLD_alpha)
            
            # Total loss = reconstruction loss (BCE) + KL divergence (KLD)
            cur_loss = BCE + KLD_alpha * KLD
            
            # Accumulate the test loss
            test_loss += cur_loss
    
    # Compute average test loss over the entire test dataset
    test_loss /= len(test_loader.dataset)
    
    # Log the average test loss
    print('====> Test set loss: {:.5f}'.format(test_loss))

    # Return the average test loss
    return test_loss

# 多级残差连接图像编码器
class ModifiedResNet(nn.Module):
    """
    A modified ResNet-inspired image encoder with multi-level residual connections.

    This model is designed for image encoding tasks where multiple residual connections are introduced
    across various layers to enhance feature learning. The architecture includes a series of convolutional
    layers with increasing and decreasing channel sizes, along with residual connections between the layers.
    The model utilizes dilated convolutions in some layers to capture broader contextual information.

    The final output is passed through a fully connected layer to generate a feature vector representation.
    """
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
    """
    CLIP (Contrastive Language-Image Pre-Training) model for learning joint image-text representations.
    
    The CLIP model learns a shared embedding space for images and texts, where corresponding image-text pairs 
    are closer together, and non-corresponding pairs are farther apart. It achieves this by encoding both 
    images and texts using separate encoders, and then computing the cosine similarity between their embeddings 
    as logits for contrastive loss.

    Args:
        text_encoder (nn.Module): The pre-trained or custom text encoder that converts text into a feature vector.
                                  This should output a feature tensor when called with text input.
    """
    def __init__(self, text_encoder):
        """
        Initializes the CLIP model, including image encoder and text encoder.

        Args:
            text_encoder (nn.Module): The text encoder that converts input text into embeddings.
        """
        super().__init__()
        self.image_encoder = ModifiedResNet()
        self.text_encoder = text_encoder   # 请确保已设置适当的参数
        
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.upsample = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=True)  # 放在 CLIP 的初始化里

    def encode_image(self, image):
        """
        Encodes the input image into a feature vector.

        Args:
            image (Tensor): The input image tensor, with shape [batch_size, channels, height, width].
        
        Returns
            Tensor: The encoded image features with shape [batch_size, feature_dim].
        """
        # image (1,36,50) -> (1,32,32) -> (32,32) -> (256,)
        image = self.upsample(image)  # 放在 encode_image 函数中
        image = image.squeeze(0)
        image_features = self.image_encoder(image)
        return image_features

    def encode_text(self, text):
        """
        Encodes the input text into a feature vector.

        Args:
            text (Tensor): The input text tensor, with shape [batch_size, text_length].
        
        Returns
            Tensor: The encoded text features with shape [batch_size, feature_dim].
        """
        # Extract text features from the text encoder (assuming it returns a tuple)
        text_features = self.text_encoder(text)[0]  # Assuming the encoder returns a tuple, we extract the first element
        return text_features


    def forward(self, image, text):
        """
        Forward pass through the CLIP model.

        Args:
            image (Tensor): The input image tensor, with shape [batch_size, channels, height, width].
            text (Tensor): The input text tensor, with shape [batch_size, text_length].
        
        Returns
            logits_per_image (Tensor): The similarity scores between images and texts, shape [batch_size, batch_size].
            logits_per_text (Tensor): The similarity scores between texts and images, shape [batch_size, batch_size].
        """
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
    """
    Trains the CLIP model with image-text pairs, using a contrastive loss function.

    This function trains the CLIP model by minimizing a cross-entropy loss on the image-text similarity. 
    The model learns to align image and text representations in a common embedding space, where the 
    image and corresponding text are closer in this space, and non-corresponding pairs are farther apart.

    Args:
        model (nn.Module): The CLIP model to be trained. It should have methods `encode_image` and `encode_text` 
                            for encoding images and texts into embeddings, respectively.
        dataloader (DataLoader): A PyTorch DataLoader object that provides batches of image-text pairs for training.
        device (torch.device): The device (CPU or GPU) on which to train the model.
        num_epochs (int): The number of epochs to train the model.

    Returns:
        tuple: A tuple containing two lists:
            - losses (list): A list of average losses per epoch.
            - accuracies (list): A list of average accuracies per epoch.
    
    Notes:
        - The optimizer used is Adam with a learning rate of 3e-4.
        - The learning rate is scheduled with a step size of 1000 and gamma of 0.1.
        - The contrastive loss is computed using cross-entropy loss between image and text logits.
        - For each image-text pair, the loss is calculated as the average cross-entropy between:
            - image logits vs. text labels
            - text logits vs. image labels
    """
    model.train()  # Set the model to training mode

    # Optimizer and learning rate scheduler
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

            # Normalize image and text embeddings
            ims = F.normalize(model.encode_image(image), dim=1)
            txt = F.normalize(model.encode_text(text), dim=1)

            # Compute image-text similarity logits
            image_logits = ims @ txt.t() * model.logit_scale.exp()
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)

            # Compute loss (average cross-entropy between image-text logits)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)) / 2

            # Compute accuracy for image and text
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()

            optimizer.zero_grad()

            # Compute image loss (again for the backward pass)
            ims = F.normalize(model.encode_image(image), dim=1)
            image_logits = ims @ txt.t() * model.logit_scale.exp()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)) / 2
            loss.backward(retain_graph=True)

            # Compute text loss
            txt = F.normalize(model.encode_text(text), dim=1)
            image_logits = ims @ txt.t() * model.logit_scale.exp()
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)) / 2
            loss.backward()

            # Update weights and learning rate
            optimizer.step()
            lr_scheduler.step()
            model.logit_scale.data.clamp_(-np.log(100), np.log(100))

            # Logging every 500 steps
            if idx % 500 == 0:
                print(f"Epoch: {epoch}, Step: {idx}, Loss: {loss.item()}, Acc: {(acc_i + acc_t) / 2 / len(image)}")

            epoch_loss += loss.item()
            epoch_acc += ((acc_i + acc_t) / 2 / len(image)).item()

            num_batches += 1

        # Calculate average loss and accuracy for this epoch
        avg_loss = epoch_loss / num_batches
        avg_acc = epoch_acc / num_batches
        losses.append(avg_loss)
        accuracies.append(avg_acc)
        print(f"Epoch: {epoch + 1}, Avg Loss: {avg_loss}, Avg Acc: {avg_acc}")

    return losses, accuracies

def clip_evaluate(model, dataloader, device):
    """
    Evaluates the CLIP model on the validation set.
    
    The `clip_evaluate` function computes the average loss and accuracy of the model 
    on a given validation dataset. It operates in evaluation mode (`model.eval()`) 
    and does not update gradients during the forward pass (`torch.no_grad()`).
    
    The evaluation is based on contrastive loss between the image and text embeddings. 
    The image and text embeddings are compared using cosine similarity, and the cross-entropy loss 
    is computed based on the similarity scores between each image-text pair.

    Args:
        model (nn.Module): The CLIP model to evaluate, which consists of image and text encoders.
        dataloader (DataLoader): A PyTorch DataLoader object containing the validation dataset.
        device (torch.device): The device (CPU or GPU) on which the model and data are loaded.
    
    Returns:
        tuple: A tuple containing the average loss and average accuracy over the entire validation set.
            - avg_loss (float): The average contrastive loss over all batches.
            - avg_acc (float): The average accuracy over all batches (measured as the average similarity 
              between image-text pairs).
    """
    model.eval()  # Set the model to evaluation mode
    epoch_loss = 0
    epoch_acc = 0
    num_batches = 0

    # No gradients needed for evaluation
    with torch.no_grad():
        for idx, (image, text) in enumerate(dataloader):
            image, text = image.to(device), text.to(device)

            # Normalize the image and text features
            ims = F.normalize(model.encode_image(image), dim=1)
            txt = F.normalize(model.encode_text(text), dim=1)

            # Compute the logits (similarities) between images and texts
            image_logits = ims @ txt.t() * model.logit_scale.exp()  # Image-to-text similarity scores
            ground_truth = torch.arange(len(image_logits)).long().to(image_logits.device)  # Ground truth for contrastive loss
            
            # Compute contrastive loss (cross-entropy loss)
            loss = (F.cross_entropy(image_logits, ground_truth) + F.cross_entropy(image_logits.t(), ground_truth)).div(2)
            
            # Compute accuracy for image-to-text and text-to-image predictions
            acc_i = (torch.argmax(image_logits, 1) == ground_truth).sum()  # Accuracy for image-to-text
            acc_t = (torch.argmax(image_logits, 0) == ground_truth).sum()  # Accuracy for text-to-image

            # Accumulate loss and accuracy for the epoch
            epoch_loss += loss.item()
            epoch_acc += ((acc_i + acc_t) / 2 / len(image)).item()
            num_batches += 1

    # Compute average loss and accuracy over all batches
    avg_loss = epoch_loss / num_batches
    avg_acc = epoch_acc / num_batches

    return avg_loss, avg_acc
