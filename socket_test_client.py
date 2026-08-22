import socket

PI_IP = "192.168.1.238"   

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((PI_IP, 9999))

client.sendall(b"Hello from laptop!")


client.close()
