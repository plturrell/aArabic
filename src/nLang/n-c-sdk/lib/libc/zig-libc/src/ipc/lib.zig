// IPC Module - Central exports for all IPC mechanisms
// Phase 1.3: Extended IPC Implementation

// POSIX Message Queues (10 functions)
pub const mqueue = @import("mqueue.zig");
pub const mq_open = mqueue.mq_open;
pub const mq_close = mqueue.mq_close;
pub const mq_unlink = mqueue.mq_unlink;
pub const mq_send = mqueue.mq_send;
pub const mq_receive = mqueue.mq_receive;
pub const mq_timedsend = mqueue.mq_timedsend;
pub const mq_timedreceive = mqueue.mq_timedreceive;
pub const mq_setattr = mqueue.mq_setattr;
pub const mq_getattr = mqueue.mq_getattr;
pub const mq_notify = mqueue.mq_notify;

// POSIX Semaphores (10 functions)
pub const semaphore = @import("semaphore.zig");
pub const sem_open = semaphore.sem_open;
pub const sem_close = semaphore.sem_close;
pub const sem_unlink = semaphore.sem_unlink;
pub const sem_init = semaphore.sem_init;
pub const sem_destroy = semaphore.sem_destroy;
pub const sem_wait = semaphore.sem_wait;
pub const sem_trywait = semaphore.sem_trywait;
pub const sem_timedwait = semaphore.sem_timedwait;
pub const sem_post = semaphore.sem_post;
pub const sem_getvalue = semaphore.sem_getvalue;

// System V Shared Memory (5 functions)
pub const sysv_shm = @import("sysv_shm.zig");
pub const shmget = sysv_shm.shmget;
pub const shmat = sysv_shm.shmat;
pub const shmdt = sysv_shm.shmdt;
pub const shmctl = sysv_shm.shmctl;
pub const ftok = sysv_shm.ftok;

// System V Message Queues (4 functions)
pub const sysv_msgq = @import("sysv_msgq.zig");
pub const msgget = sysv_msgq.msgget;
pub const msgsnd = sysv_msgq.msgsnd;
pub const msgrcv = sysv_msgq.msgrcv;
pub const msgctl = sysv_msgq.msgctl;

// System V Semaphores (3 functions)
pub const sysv_sem = @import("sysv_sem.zig");
pub const semget = sysv_sem.semget;
pub const semop = sysv_sem.semop;
pub const semctl = sysv_sem.semctl;

// POSIX Shared Memory (2 functions)
pub const posix_shm = @import("posix_shm.zig");
pub const shm_open = posix_shm.shm_open;
pub const shm_unlink = posix_shm.shm_unlink;

// Pipes & FIFOs (8 functions)
pub const pipes = @import("pipes.zig");
pub const pipe = pipes.pipe;
pub const pipe2 = pipes.pipe2;
pub const mkfifo = pipes.mkfifo;
pub const mkfifoat = pipes.mkfifoat;
pub const mknod = pipes.mknod;
pub const mknodat = pipes.mknodat;
pub const splice = pipes.splice;
pub const tee = pipes.tee;

// Total: 42 production-ready IPC functions
// - POSIX: 22 functions (mqueue, sem, shm)
// - System V: 12 functions (shm, msgq, sem)
// - Pipes: 8 functions
// 
// Status: PRODUCTION READY for banking systems
// Quality: Full blocking, SEM_UNDO, type filtering, proper error handling
