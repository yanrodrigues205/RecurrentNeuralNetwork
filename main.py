# main.py

import numpy as np
import matplotlib.pyplot as plt
from data_preprocessador import load_data
from model import create_model, train_model

def main():
    filename = 'stock_prices.csv'  # Substitua pelo seu arquivo de dados
    sequence_length = 5  # Define um comprimento de sequência menor para o exemplo
    
    X, y, scaler = load_data(filename, sequence_length=sequence_length)

    # Dividir os dados em conjuntos de treinamento e teste
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    input_shape = (X_train.shape[1], 1)
    model = create_model(input_shape)

    model, history = train_model(model, X_train, y_train)

    # Avaliar o modelo no conjunto de teste
    loss = model.evaluate(X_test, y_test)
    print(f'Test Loss: {loss}')
    
    # Plotar a perda do treinamento
    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.show()

    # Salvar o modelo treinado
    model.save('trained_lstm_model.h5')

if __name__ == "__main__":
    main()
