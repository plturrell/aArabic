#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <regex.h>
#include <netdb.h>
#include <signal.h>
#include <unistd.h>
#include <math.h>
#include <errno.h>

// Global counter for threads
int counter = 0;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

void* thread_func(void* arg) {
    pthread_mutex_lock(&lock);
    counter++;
    printf("Thread %ld incremented counter to %d\n", (long)pthread_self(), counter);
    pthread_mutex_unlock(&lock);
    return NULL;
}

void signal_handler(int sig) {
    printf("Caught signal %d\n", sig);
}

int main() {
    printf("=== zig-libc Integration Test ===\n");

    // 1. Math
    double s = sin(1.57079632679); // pi/2
    printf("sin(pi/2) = %f (expected ~1.0)\n", s);
    if (fabs(s - 1.0) > 0.001) {
        printf("FAIL: Math sin() error\n");
        return 1;
    }

    // 2. Signals
    printf("Testing signals...\n");
    signal(SIGUSR1, signal_handler);
    raise(SIGUSR1);

    // 3. Threads
    printf("Testing pthreads...\n");
    pthread_t t1, t2;
    pthread_create(&t1, NULL, thread_func, NULL);
    pthread_create(&t2, NULL, thread_func, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    printf("Final counter: %d (expected 2)\n", counter);
    if (counter != 2) {
        printf("FAIL: Thread counter mismatch\n");
        return 1;
    }

    // 4. Regex
    printf("Testing regex...\n");
    regex_t regex;
    int reti = regcomp(&regex, "^[a-z0-9]+$", 0);
    if (reti) {
        printf("FAIL: regcomp\n");
        return 1;
    }
    reti = regexec(&regex, "hello123", 0, NULL, 0);
    if (!reti) {
        printf("Regex match: success\n");
    } else {
        printf("FAIL: Regex match failed\n");
        return 1;
    }
    regfree(&regex);

    // 5. NetDB (/etc/hosts)
    printf("Testing gethostbyname (localhost)...");
    struct hostent* h = gethostbyname("localhost");
    if (h) {
        printf("Resolved localhost: %s\n", h->h_name);
    } else {
        printf("WARN: gethostbyname(localhost) failed (check /etc/hosts)\n");
    }

    // 6. Stdio (File I/O)
    printf("Testing file I/O...\n");
    FILE* f = fopen("test.txt", "w");
    if (!f) {
        printf("FAIL: fopen\n");
        return 1;
    }
    fprintf(f, "Hello zig-libc!");
    fclose(f);

    f = fopen("test.txt", "r");
    char buf[100];
    fgets(buf, 100, f);
    fclose(f);
    printf("Read from file: %s\n", buf);
    if (strcmp(buf, "Hello zig-libc!") != 0) {
        printf("FAIL: File content mismatch\n");
        return 1;
    }
    remove("test.txt");

    printf("=== All tests passed ===\n");
    return 0;
}
