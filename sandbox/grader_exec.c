#define _GNU_SOURCE

#include <errno.h>
#include <grp.h>
#include <limits.h>
#include <seccomp.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static void die(const char *message) {
    perror(message);
    exit(125);
}

static void set_limit(int resource, rlim_t value) {
    struct rlimit limit = {.rlim_cur = value, .rlim_max = value};
    if (setrlimit(resource, &limit) != 0) {
        die("setrlimit");
    }
}

static void deny_syscall(scmp_filter_ctx context, const char *name) {
    int syscall_number = seccomp_syscall_resolve_name(name);
    if (syscall_number == __NR_SCMP_ERROR) {
        return;
    }
    if (seccomp_rule_add(
            context, SCMP_ACT_ERRNO(EPERM), syscall_number, 0) != 0) {
        fprintf(stderr, "failed to restrict syscall: %s\n", name);
        exit(125);
    }
}

static void install_filter(void) {
    const char *denied[] = {
        /* Network access. */
        "socket", "socketpair", "connect", "bind", "listen", "accept",
        "accept4", "sendto", "sendmsg", "sendmmsg", "recvfrom", "recvmsg",
        "recvmmsg", "shutdown", "setsockopt", "getsockopt",

        /* Namespace, mount, kernel, and cross-process escape surfaces. */
        "mount", "umount2", "pivot_root", "move_mount", "open_tree",
        "fsopen", "fsconfig", "fsmount", "fspick", "mount_setattr",
        "unshare", "setns", "ptrace", "process_vm_readv",
        "process_vm_writev", "open_by_handle_at", "name_to_handle_at", "bpf",
        "perf_event_open", "userfaultfd", "io_uring_setup", "io_uring_enter",
        "io_uring_register", "keyctl", "add_key", "request_key", "reboot",
        "kexec_load", "kexec_file_load", "init_module", "finit_module",
        "delete_module", "swapon", "swapoff", "acct", "quotactl",

        /* Keep each grader process group controllable by the parent. */
        "setsid", "setpgid", "kill", "tkill", "tgkill",
        "pidfd_send_signal",

        /* Avoid cross-job System V IPC shared by the same unprivileged UID. */
        "shmget", "shmat", "shmdt", "shmctl", "semget", "semop",
        "semtimedop", "semctl", "msgget", "msgsnd", "msgrcv", "msgctl",
    };

    scmp_filter_ctx context = seccomp_init(SCMP_ACT_ALLOW);
    if (context == NULL) {
        die("seccomp_init");
    }
    for (size_t index = 0; index < sizeof(denied) / sizeof(denied[0]); index++) {
        deny_syscall(context, denied[index]);
    }
    if (seccomp_load(context) != 0) {
        seccomp_release(context);
        die("seccomp_load");
    }
    seccomp_release(context);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: grader_exec ROOTFS /work/JOB_DIRECTORY\n");
        return 125;
    }
    const char *rootfs = argv[1];
    const char *workdir = argv[2];
    if (rootfs[0] != '/' || strncmp(workdir, "/work/", 6) != 0 ||
        strstr(workdir, "..") != NULL) {
        fprintf(stderr, "invalid rootfs or work directory\n");
        return 125;
    }

    set_limit(RLIMIT_CPU, 12);
    set_limit(RLIMIT_AS, 1ULL * 1024ULL * 1024ULL * 1024ULL);
    set_limit(RLIMIT_FSIZE, 1024ULL * 1024ULL);
    set_limit(RLIMIT_NPROC, 16);
    set_limit(RLIMIT_NOFILE, 64);
    set_limit(RLIMIT_CORE, 0);

    if (chdir(rootfs) != 0 || chroot(".") != 0 || chdir(workdir) != 0) {
        die("chroot/chdir");
    }

    long open_max = sysconf(_SC_OPEN_MAX);
    if (open_max < 0 || open_max > 65536) {
        open_max = 65536;
    }
    for (int fd = 3; fd < open_max; fd++) {
        close(fd);
    }

    if (setgroups(0, NULL) != 0 || setgid(65534) != 0 || setuid(65534) != 0) {
        die("drop privileges");
    }
    umask(077);
    if (prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0) {
        die("prctl(PR_SET_DUMPABLE)");
    }
    if (prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0) {
        die("prctl(PR_SET_NO_NEW_PRIVS)");
    }
    install_filter();

    char *const child_argv[] = {
        "/usr/bin/python3", "-E", "-s", "-B", "-m", "pytest",
        "--capture=sys", "-p", "no:cacheprovider", "-p", "no:logging",
        "-q", "test_solution.py", NULL,
    };
    char *const child_env[] = {
        "PATH=/usr/bin:/bin",
        "HOME=/nonexistent",
        "LANG=C.UTF-8",
        "LC_ALL=C.UTF-8",
        "PYTHONDONTWRITEBYTECODE=1",
        "PYTHONUNBUFFERED=1",
        "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1",
        NULL,
    };
    execve(child_argv[0], child_argv, child_env);
    die("execve python3");
    return 125;
}
