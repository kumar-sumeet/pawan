#ifndef NETWORKINTERFACE_H
#define NETWORKINTERFACE_H

#if defined(__WIN32__)
#include <io.h>
#include <winsock2.h>
#endif
#if defined(__linux__)
#include <arpa/inet.h>
#include <cstdlib>
#include <cstring>
#include <errno.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#define SOCKET int
#define INVALID_SOCKET -1
#define SOCKET_ERROR -1
#endif

#include <iostream>
#include <stdio.h>

#if defined(__WIN32__)
#pragma comment(lib, "ws2_32.lib") // Winsock Library

int winsock_init(WSADATA &wsa)
{

  if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0)
  {
    printf("Failed. Error Code : %d \n", WSAGetLastError());
    return 1;
  }

  return 0;
};
#endif

#define CHUNKSIZE 1400

/*
Class to manage network communication
*/

template <class DATA_IN, class DATA_OUT>
class NetworkInterface
{

protected:
    // basic variables
#if defined(__WIN32__)
    WSADATA _wsa;
#endif
    SOCKET _socket;
    unsigned short _buffermultiplicator;
    // server ports and data
    unsigned short _in_port;
    const char *_out_ip;
    unsigned short _out_port;
    // pointers to buffers
    void *_recieveBuffer;
    void *_sendBuffer;
    int recieved = 0;

    // sockets to communicate
    SOCKET _in_socket, _out_socket;
    struct sockaddr_in _in_addr, _out_addr;
    unsigned int _sizeofaddr;

    NetworkInterface() = delete;

    NetworkInterface(
            unsigned short in_port, const char *out_ip, unsigned short out_port,
            unsigned short
            buffermultiplicator); // creates everything up to after the connection

    virtual ~NetworkInterface(); // closes sockets and undloads windows dll

public:
    virtual void recieve_data(DATA_IN &data) = 0;
    virtual void getrecieveBuffer(DATA_IN &data) = 0;

    virtual void send_data(DATA_OUT &data) = 0;

    void recreate_socket();

    virtual int socket_init() = 0;
};

template <class DATA_IN, class DATA_OUT>
class NetworkInterfaceTCP : public NetworkInterface<DATA_IN, DATA_OUT>
{

protected:
    bool _server_mode;

public:
    NetworkInterfaceTCP() = delete;

    NetworkInterfaceTCP(unsigned short in_port, const char *out_ip,
                        unsigned short out_port,
                        unsigned short buffermultiplicator,
                        bool server_mode = false);

    void recieve_data(DATA_IN &data);
    void getrecieveBuffer(DATA_IN &data);
    void send_data(DATA_OUT &data);

    void reconnect();

    int socket_init();
};

template <class DATA_IN, class DATA_OUT>
class NetworkInterfaceUDP : public NetworkInterface<DATA_IN, DATA_OUT>
{

public:
    NetworkInterfaceUDP() = delete;

    NetworkInterfaceUDP(unsigned short in_port, const char *out_ip,
                        unsigned short out_port,
                        unsigned short buffermultiplicator);

    void recieve_data(DATA_IN &data);

    void send_data(DATA_OUT &data);

    int socket_init();
};
#endif

