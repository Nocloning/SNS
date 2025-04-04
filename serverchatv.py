import socket
from forecast import predict

pre = predict("AAPL")
print("prediction:", pre)


host_ip = socket.gethostbyname(socket.gethostname())  # dynamically gets ip address of local machine,
# can use localhost lookpack 127.0.0.1
host_port = 61825 # non-privileged ports for the client to listen on can be used > 1023
encoder = "utf-8"  # decodes and encodes strings entered
byte = 1024 # 1024 bytes at a time are sent over a connection socket

#a server socket is created server_socket using the function socket.socket()
#accepts AF_INET internet address family of IPv4 address, and  SOCK_STREAM
# which is a is a socket type TCP , UDP types are possible as well
# bind() function binds socket to to host and port number
# and listen() function is used to listen to the ports for possible client request
# after connection, a connection is specified by connection port and addresss of the connection

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((host_ip, host_port))
server_socket.listen()

print("Server Listening...\n")

client_socket, client_address = server_socket.accept()

client_socket.send("Connected to Chatbot S&P500 forecaster...\n".encode(encoder))

# we create a loop when a client is connected and send mesaages back and forth
# uses recv() and sendall() functions to read and send clients data
# exit the loop when a client types exit
# close client socket when we break out of the loop
while True:
    data = client_socket.recv(byte).decode(encoder)
  #  print("Server Message",data)
    if data == "exit":
        client_socket.send("exit".encode(encoder))
        print("\nExiting...bye for now")
        break
    else:
        print("Client:" + data)
        data = input("Chat:")
        client_socket.sendall(data.encode(encoder))

pre = predict("ticker")
print("prediction:", pre)

server_socket.close()
