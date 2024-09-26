import serial
import csv
import re

# Configurações da porta serial
serial_port = '/dev/ttyACM0'  # Altere para sua porta serial
baud_rate = 115200  # Taxa de comunicação
csv_file = 'valores_saida.csv'  # Nome do arquivo CSV para salvar os dados

# Expressão regular para capturar os valores separados por ponto e vírgula
pattern = re.compile(r'([\d\.]+);([\d\.]+);([\d\.]+);([\d]+)')

# Função para iniciar a leitura da porta serial
def ler_dados_serial():
    with serial.Serial(serial_port, baud_rate, timeout=1) as ser, open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Na', 'K', 'Cl', 'Inference_Time'])  # Cabeçalhos do CSV

        lendo_dados = False  # Flag para saber se devemos processar os dados

        while True:
            linha = ser.readline().decode('utf-8').strip()  # Lê uma linha do terminal e remove espaços extras
            if linha:
                print(f"Recebido: {linha}")

                # Verifica se a leitura deve começar
                if "Modelo configurado com sucesso." in linha:
                    lendo_dados = True
                    print("Iniciando captura de dados...")

                # Verifica se deve parar a leitura
                if "encerrados dados de teste!" in linha:
                    print("Encerrando captura de dados...")
                    break

                # Se estamos lendo dados, capturar as linhas com valores
                if lendo_dados:
                    match = pattern.match(linha)
                    if match:
                        na_value = float(match.group(1))
                        k_value = float(match.group(2))
                        cl_value = float(match.group(3))
                        inference_time = int(match.group(4))
                        
                        # Escreve os valores no arquivo CSV
                        writer.writerow([na_value, k_value, cl_value, inference_time])
                        print(f"Valores capturados: Na={na_value}, K={k_value}, Cl={cl_value}, Tempo={inference_time} µs")

if __name__ == "__main__":
    ler_dados_serial()