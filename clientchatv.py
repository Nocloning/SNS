import socket


dest_ip = socket.gethostbyname(socket.gethostname())  # dynamically ip address of local machine,
# can use localhost lookpack 127.0.0.1
dest_port = 61825 # non-privileged ports for the client to connect to server
encoder = "utf-8"  # decodes and encodes strings entered
byte = 1024 # 1024 bytes at a time are sent over a connection socket


#a server socket is created server_socket using the function socket.socket()
#accepts AF_INET internet address family of IPv4 address, and  SOCK_STREAM
# which is a is a socket type TCP , UDP types are possible as well

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((dest_ip, dest_port))


# we create a loop when connection is made
# uses recv() and sendall() functions to read and send clients data, uses char.isdigit() to check
# for non digit entries and rejects commands user to enter correct ticker symbol
# exit the loop when exit is received
# close client socket when we break out of the loop

while True:
    data = client_socket.recv(byte).decode(encoder)
    #print(data)
    if data == "exit":
        client_socket.sendall("exit".encode(encoder))
        print("\nExiting...")
        break
    else:
        print("Oracle:" + data)
        data = input("Enter Ticker from SNP500 for forecast:")
        if any(char.isdigit() for char in data):
            print("Only ticker characters no integers:" + data)
        client_socket.sendall(data.encode(encoder))

client_socket.close()