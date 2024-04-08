#ifndef NETWORKINTERFACE_CPP
#define NETWORKINTERFACE_CPP

#include "networkinterface.h"
#include <chrono>
#include <thread>

/// Definition of parent class
template <class DATA_IN, class DATA_OUT>
NetworkInterface<DATA_IN, DATA_OUT>::NetworkInterface(
        unsigned short in_port, const char *out_ip, unsigned short out_port,
        unsigned short buffermultiplicator)
        :

        _in_port(in_port)
        , _out_ip(out_ip)
        , _out_port(out_port)
        , _buffermultiplicator(buffermultiplicator)
        , _recieveBuffer(new char[sizeof(DATA_IN) * _buffermultiplicator])
        , _sendBuffer(new char[sizeof(DATA_OUT) * _buffermultiplicator])

{
#if defined(__WIN32__)
    winsock_init(_wsa);
#endif
};

template <class DATA_IN, class DATA_OUT>
NetworkInterface<DATA_IN, DATA_OUT>::~NetworkInterface()
{

    delete[] static_cast<char *>(_recieveBuffer);
    delete[] static_cast<char *>(_sendBuffer);

#if defined(__WIN32__)
    closesocket(_socket);
  WSACleanup();
#endif
#if defined(__linux__)
    close(_socket);
#endif
};

template <class DATA_IN, class DATA_OUT>
void NetworkInterface<DATA_IN, DATA_OUT>::recreate_socket()
{

#if defined(__WIN32__)
    closesocket(_socket);
  WSACleanup();
#endif
#if defined(__linux__)
    std::cout << "Closing socket" << std::endl;
    int success = close(_socket);
    std::cout << "Result: " << success << std::endl;
#endif

    std::cout << "Recreating socket" << std::endl;

#if defined(__WIN32__)
    winsock_init(_wsa);
#endif
    int ret = socket_init();
    std::cout << "socket init returned " << ret << std::endl;
}

/// Definition of child for TCP connections
template <class DATA_IN, class DATA_OUT>
NetworkInterfaceTCP<DATA_IN, DATA_OUT>::NetworkInterfaceTCP(
        unsigned short in_port, const char *out_ip, unsigned short out_port,
        unsigned short buffermultiplicator, bool server_mode)
        : NetworkInterface<DATA_IN, DATA_OUT>(in_port, out_ip, out_port,
                                              buffermultiplicator)
        , _server_mode(server_mode)

{};

template <class DATA_IN, class DATA_OUT>
int NetworkInterfaceTCP<DATA_IN, DATA_OUT>::socket_init()
{

    // create new socket (AF_INET for ipv4, SOCK_STREAM for TCP)
    if ((this->_socket = socket(AF_INET, SOCK_STREAM, 0)) == INVALID_SOCKET)
    {
#if defined(__WIN32__)
        printf("Could not create socket : %d \n", WSAGetLastError());
#else
        printf("Could not create socket %s\n", strerror(errno));
#endif
        return 1;
    }

    if (_server_mode)
    {

        // Prepare the structure
        this->_in_addr.sin_family = AF_INET;
        this->_in_addr.sin_addr.s_addr = INADDR_ANY;
        this->_in_addr.sin_port = htons(this->_in_port);

        int enable = 1;
        if (setsockopt(this->_socket, SOL_SOCKET, SO_REUSEADDR, &enable,
                       sizeof(int)) < 0)
            std::cout << "setsockopt(SO_REUSEADDR) failed" << std::endl;
        // Bind
        /*if( bind(_socket ,(struct sockaddr *)&_in_addr , sizeof(_in_addr)) != 0)
        {
                printf("Bind failed with error code : %d" , WSAGetLastError());
                exit(EXIT_FAILURE);
        }*/
        while (true)
        {
            if (bind(this->_socket, (struct sockaddr *)&this->_in_addr,
                     sizeof(this->_in_addr)) != 0)
            {

#if defined(__WIN32__)
                printf("Binding of socket failed %d sleeping for 30secs\n",
               WSAGetLastError());
#else
                printf("Binding of socket failed %s sleeping for 30secs\n",
                       strerror(errno));
#endif
                std::this_thread::sleep_for(std::chrono::milliseconds(30000));
            }
            else
                break;
        }

        // Listen to incoming connections
        listen(this->_socket, this->_buffermultiplicator);

        this->_sizeofaddr = sizeof(struct sockaddr_in);

        printf("Server listening for connections \n");

        if ((this->_in_socket =
                     accept(this->_socket, (struct sockaddr *)&this->_out_addr,
                            &this->_sizeofaddr)) != INVALID_SOCKET)
        {
            printf("Connection on port %u accepted \n",
                   ntohs(this->_in_addr.sin_port));

            printf("Replying to %d.%d.%d.%d:%u \n",
                   int(this->_out_addr.sin_addr.s_addr & 0xFF),
                   int((this->_out_addr.sin_addr.s_addr & 0xFF00) >> 8),
                   int((this->_out_addr.sin_addr.s_addr & 0xFF0000) >> 16),
                   int((this->_out_addr.sin_addr.s_addr & 0xFF000000) >> 24),
                   ntohs(this->_out_addr.sin_port));
        }

        else
        {

#if defined(__WIN32__)
            printf("accept failed with error code : %d \n", WSAGetLastError());
#else
            printf("accept failed with error code : %s\n", strerror(errno));
#endif
            return 1;
        }

        // take accepted socket (remote) as out socket
        this->_out_socket = this->_in_socket;
    }
    else
    {

        // Prepare the structure
        this->_out_addr.sin_family = AF_INET;
        this->_out_addr.sin_addr.s_addr = inet_addr(this->_out_ip);
        this->_out_addr.sin_port = htons(this->_out_port);

        if (connect(this->_socket, (struct sockaddr *)&this->_out_addr,
                    sizeof(this->_out_addr)) < 0)
        {
#if defined(__WIN32__)
            printf("Could not connect to server : %d \n", WSAGetLastError());
#else
            printf("Could not connect to server : %s\n", strerror(errno));
#endif
            return 1;
        }
        else
        {
            printf("Connected to server %d.%d.%d.%d:%u \n",
                   int(this->_out_addr.sin_addr.s_addr & 0xFF),
                   int((this->_out_addr.sin_addr.s_addr & 0xFF00) >> 8),
                   int((this->_out_addr.sin_addr.s_addr & 0xFF0000) >> 16),
                   int((this->_out_addr.sin_addr.s_addr & 0xFF000000) >> 24),
                   ntohs(this->_out_addr.sin_port));
            this->_out_socket = this->_socket;
            this->_in_socket = this->_socket;
        }
    }

    // std::cout << " end of socket init" << std::endl;
    return 0;
};

template <class DATA_IN, class DATA_OUT>
void NetworkInterfaceTCP<DATA_IN, DATA_OUT>::reconnect()
{
    listen(this->_socket, this->_buffermultiplicator);

    this->_sizeofaddr = sizeof(struct sockaddr_in);

    printf("Server listening for connections \n");

    if ((this->_in_socket =
                 accept(this->_socket, (struct sockaddr *)&this->_out_addr,
                        &this->_sizeofaddr)) != INVALID_SOCKET)
    {
        printf("Connection on port %u accepted \n", ntohs(this->_in_addr.sin_port));

        printf("Replying to %d.%d.%d.%d:%u \n",
               int(this->_out_addr.sin_addr.s_addr & 0xFF),
               int((this->_out_addr.sin_addr.s_addr & 0xFF00) >> 8),
               int((this->_out_addr.sin_addr.s_addr & 0xFF0000) >> 16),
               int((this->_out_addr.sin_addr.s_addr & 0xFF000000) >> 24),
               ntohs(this->_out_addr.sin_port));
    }

    else
    {
#if defined(__WIN32__)
        printf("accept failed with error code : %d", WSAGetLastError());
#else
        printf("accept failed");
#endif
        return;
    }

    // take accepted socket (remote) as out socket
    this->_out_socket = this->_in_socket;

    return;
}

template <class DATA_IN, class DATA_OUT>
void NetworkInterfaceTCP<DATA_IN, DATA_OUT>::recieve_data(DATA_IN &data)
{

    int Already_recvt = {0};
    int recvt = {0};

    while (Already_recvt != sizeof(DATA_IN))
    {

        recvt = recv(this->_in_socket, /*(char *)*/
                     (static_cast<char *>(this->_recieveBuffer) +
                      Already_recvt /*+i*CHUNKSIZE*/),
                     sizeof(DATA_IN), 0);
        Already_recvt += recvt;
    }
    printf("I received %i bytes, should be %lu\n",Already_recvt,sizeof(DATA_IN));
    std::memcpy(&data, this->_recieveBuffer, sizeof(DATA_IN));
};

template <class DATA_IN, class DATA_OUT>
void NetworkInterfaceTCP<DATA_IN, DATA_OUT>::getrecieveBuffer(DATA_IN &data)
{
    std::memcpy(&data, this->_recieveBuffer, sizeof(DATA_IN));
};

template <class DATA_IN, class DATA_OUT>
void NetworkInterfaceTCP<DATA_IN, DATA_OUT>::send_data(DATA_OUT &data)
{
    // int a;
    int Already_sent = {0};

    std::memcpy(this->_sendBuffer, &data, sizeof(DATA_OUT));

    Already_sent +=
            send(this->_out_socket, (char *)(this->_sendBuffer /*+(i)*CHUNKSIZE*/),
                 sizeof(DATA_OUT), 0);

    if (Already_sent != sizeof(DATA_OUT))
    {
        std::cout << "Bytes Send " << Already_sent << " Size of total package "
                  << sizeof(DATA_OUT) << std::endl;
    }

    // std::cout << "Sent " << Already_sent << " bytes, should be " <<
    // sizeof(DATA_OUT) << std::endl;
};

/// Definition of child for UDP connections
template <class DATA_IN, class DATA_OUT>
NetworkInterfaceUDP<DATA_IN, DATA_OUT>::NetworkInterfaceUDP(
        unsigned short in_port, const char *out_ip, unsigned short out_port,
        unsigned short buffermultiplicator)
        : NetworkInterface<DATA_IN, DATA_OUT>(in_port, out_ip, out_port,
                                              buffermultiplicator)

{};

template <class DATA_IN, class DATA_OUT>
int NetworkInterfaceUDP<DATA_IN, DATA_OUT>::socket_init()
{

    // create new socket (AF_INET for ipv4, SOCK_STREAM for TCP)
    if ((this->_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) ==
        INVALID_SOCKET)
    {
#if defined(__WIN32__)
        printf("Could not create socket : %d \n", WSAGetLastError());
#else
        printf("Could not create socket");
#endif
        return 1;
    }

    // Prepare the structure
    this->_in_addr.sin_family = AF_INET;
    this->_in_addr.sin_addr.s_addr = INADDR_ANY;
    this->_in_addr.sin_port = htons(this->_in_port);

    // Bind
    if (bind(this->_socket, (struct sockaddr *)&this->_in_addr,
             sizeof(this->_in_addr)) == SOCKET_ERROR)
    {
#if defined(__WIN32__)
        printf("Bind failed with error code : %d", WSAGetLastError());
#else
        printf("Bind failed with error code ");
#endif
        exit(EXIT_FAILURE);
    }
    else
        printf("Socket with port %u opened for incoming data via UDP \n",
               ntohs(this->_in_addr.sin_port));

    // pass socket to in_socket for easier handling
    this->_in_socket = this->_socket;

    // opening socket for outgoing connections
    if ((this->_out_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP)) < 0)
    {
        printf("Socket failed\n"); /* return false; */
    }

    // memset((char *) &_out_addr, 0, sizeof(_out_addr));
    this->_out_addr.sin_family = AF_INET;
    this->_out_addr.sin_addr.s_addr = inet_addr(this->_out_ip);
    this->_out_addr.sin_port = htons(this->_out_port);

    //	if( bind(_out_socket ,(struct sockaddr *)&_out_addr , sizeof(_out_addr))
    //== SOCKET_ERROR)
    //    {
    //        printf("Bind failed with error code : %d" , WSAGetLastError());
    //        exit(EXIT_FAILURE);
    //    }
    //	else printf("Socket with port %u opened for outgoing data via UDP \n",
    //ntohs(_out_addr.sin_port) );

    return 0;
};

template <class DATA_IN, class DATA_OUT>
void NetworkInterfaceUDP<DATA_IN, DATA_OUT>::recieve_data(DATA_IN &data)
{
    // std::cout << "call" << std::endl;
    int recvreturn = 0;
    int count = 0;
    bool gotsomething = false;
    // std::cout << "nach decl, size _recieveBuffer " << sizeof(_recieveBuffer) <<
    // " theor size " << sizeof(DATA_OUT)*_buffermultiplicator << std::endl;
    while (!gotsomething)
    {
        // std::cout << "Count " << count << std::endl;
        do
        {
            recvreturn = recvfrom(
                    this->_in_socket, (char *)this->_recieveBuffer,
                    sizeof(data) * this->_buffermultiplicator, 0, NULL,
                    0); // size may be wrong (sockaddr *)_in_addr, sizeof(_in_addr)
            // std::cout << "value in recvreturn " << recvreturn << std::endl;
            if (recvreturn == SOCKET_ERROR)
            {
#if defined(__WIN32__)
                wprintf(L"sendto failed with error: %d\n", WSAGetLastError());
#else
                wprintf(L"sendto failed \n");
#endif
            }
            if (recvreturn > 0) // if valid data was found copy it to the readbuffer
            {
                // for(i=0;i<sizeof(_recieveBuffer);i++)
                // readBuffer[i]=_recieveBuffer[i];
                std::memcpy(&data, this->_recieveBuffer, sizeof(data));
                gotsomething = true;
            }
            // std::cout << "inner loop" << std::endl;
        } while (recvreturn > 0); // ends either at once or when there's no more
        // valid data on the buffer
        ++count;
    }
};

template <class DATA_IN, class DATA_OUT>
void NetworkInterfaceUDP<DATA_IN, DATA_OUT>::send_data(DATA_OUT &data)
{

    int retval;
    std::memcpy(this->_sendBuffer, &data, sizeof(data));

    retval = sendto(
            this->_out_socket, (char *)this->_sendBuffer, sizeof(data), 0,
            (struct sockaddr *)&this->_out_addr,
            sizeof(this->_out_addr)); //(sockaddr *)_out_addr, sizeof(_out_addr)

    if (retval == SOCKET_ERROR)
    {
#if defined(__WIN32__)
        wprintf(L"sendto failed with error: %d\n", WSAGetLastError());
#else
        wprintf(L"sendto failed with error:\n");
#endif
    }
};

#endif
