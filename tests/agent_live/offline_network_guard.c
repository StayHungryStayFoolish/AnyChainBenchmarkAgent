#define _GNU_SOURCE

#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <stddef.h>
#include <sys/socket.h>
#include <sys/types.h>

static int address_is_loopback(const struct sockaddr *address, socklen_t length) {
    if (address == NULL) {
        return 1;
    }
    if (address->sa_family == AF_UNIX) {
        return 1;
    }
    if (address->sa_family == AF_INET && length >= sizeof(struct sockaddr_in)) {
        const struct sockaddr_in *ipv4 = (const struct sockaddr_in *)address;
        return (ntohl(ipv4->sin_addr.s_addr) >> 24) == 127;
    }
    if (address->sa_family == AF_INET6 && length >= sizeof(struct sockaddr_in6)) {
        const struct sockaddr_in6 *ipv6 = (const struct sockaddr_in6 *)address;
        return IN6_IS_ADDR_LOOPBACK(&ipv6->sin6_addr);
    }
    return 0;
}

int connect(int socket_fd, const struct sockaddr *address, socklen_t length) {
    static int (*real_connect)(int, const struct sockaddr *, socklen_t) = NULL;
    if (!address_is_loopback(address, length)) {
        errno = ENETUNREACH;
        return -1;
    }
    if (real_connect == NULL) {
        real_connect = dlsym(RTLD_NEXT, "connect");
    }
    return real_connect(socket_fd, address, length);
}

ssize_t sendto(
    int socket_fd,
    const void *buffer,
    size_t size,
    int flags,
    const struct sockaddr *destination,
    socklen_t length
) {
    static ssize_t (*real_sendto)(
        int,
        const void *,
        size_t,
        int,
        const struct sockaddr *,
        socklen_t
    ) = NULL;
    if (!address_is_loopback(destination, length)) {
        errno = ENETUNREACH;
        return -1;
    }
    if (real_sendto == NULL) {
        real_sendto = dlsym(RTLD_NEXT, "sendto");
    }
    return real_sendto(
        socket_fd,
        buffer,
        size,
        flags,
        destination,
        length
    );
}

ssize_t sendmsg(int socket_fd, const struct msghdr *message, int flags) {
    static ssize_t (*real_sendmsg)(int, const struct msghdr *, int) = NULL;
    if (
        message != NULL
        && message->msg_name != NULL
        && !address_is_loopback(
            (const struct sockaddr *)message->msg_name,
            (socklen_t)message->msg_namelen
        )
    ) {
        errno = ENETUNREACH;
        return -1;
    }
    if (real_sendmsg == NULL) {
        real_sendmsg = dlsym(RTLD_NEXT, "sendmsg");
    }
    return real_sendmsg(socket_fd, message, flags);
}
