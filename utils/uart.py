import serial
import csv
import re

serial_port = '/dev/ttyACM1'
baud_rate = 115200
csv_file = 'valores_saida.csv'

pattern = re.compile(r'Na:([\d\.]+)\s+K:([\d\.]+)\s+Cl:([\d\.]+)\s+Inf\.Time:(\d+)µs')

def ler_dados_serial():
    try:
        with serial.Serial(serial_port, baud_rate, timeout=1) as ser, open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Na', 'K', 'Cl', 'Inference_Time'])

            lendo_dados = False

            while True:
                linha = ser.readline().decode('utf-8').strip()
                if linha:
                    print(f"Recebido: {linha}")

                    if "Modelo configurado com sucesso." in linha:
                        lendo_dados = True
                        print("Iniciando captura de dados...")

                    if "Encerrados dados de teste!" in linha:
                        print("Encerrando captura de dados...")
                        break

                    if lendo_dados:
                        match = pattern.search(linha)
                        if match:
                            try:
                                na_value = float(match.group(1))
                                k_value = float(match.group(2))
                                cl_value = float(match.group(3))
                                inference_time = int(match.group(4))
                                
                                writer.writerow([na_value, k_value, cl_value, inference_time])
                                print(f"Valores capturados: Na={na_value}, K={k_value}, Cl={cl_value}, Tempo={inference_time} µs")
                            except ValueError as e:
                                print(f"Erro ao converter valores: {e}")
                        else:
                            print("Linha não corresponde ao padrão esperado.")
    except serial.SerialException as e:
        print(f"Erro de comunicação serial: {e}")
    except IOError as e:
        print(f"Erro de escrita no arquivo CSV: {e}")

if __name__ == "__main__":
    ler_dados_serial()
