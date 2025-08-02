#!/bin/bash
# Security hardening script for Docker containers and deployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -eq 0 ]]; then
        print_warning "Running as root. Consider using a non-root user for better security."
    fi
}

# Function to harden Docker daemon configuration
harden_docker_daemon() {
    print_status "Hardening Docker daemon configuration..."
    
    # Check if Docker daemon configuration directory exists
    if [[ ! -d "/etc/docker" ]]; then
        sudo mkdir -p /etc/docker
    fi
    
    # Create or update daemon.json with security settings
    cat <<EOF | sudo tee /etc/docker/daemon.json > /dev/null
{
    "icc": false,
    "userns-remap": "default",
    "no-new-privileges": true,
    "seccomp-profile": "/etc/docker/seccomp/default.json",
    "apparmor-profile": "docker-default",
    "selinux-enabled": true,
    "disable-legacy-registry": true,
    "live-restore": true,
    "userland-proxy": false,
    "experimental": false,
    "log-driver": "json-file",
    "log-opts": {
        "max-size": "10m",
        "max-file": "3"
    },
    "storage-driver": "overlay2",
    "storage-opts": [
        "overlay2.override_kernel_check=true"
    ]
}
EOF
    
    print_success "Docker daemon configuration updated"
}

# Function to set up container security profiles
setup_security_profiles() {
    print_status "Setting up container security profiles..."
    
    # Create AppArmor profile directory
    if [[ ! -d "/etc/apparmor.d/docker" ]]; then
        sudo mkdir -p /etc/apparmor.d/docker
    fi
    
    # Create custom seccomp profile
    cat <<EOF | sudo tee /etc/docker/seccomp/sql-synth-profile.json > /dev/null
{
    "defaultAction": "SCMP_ACT_ERRNO",
    "archMap": [
        {
            "architecture": "SCMP_ARCH_X86_64",
            "subArchitectures": [
                "SCMP_ARCH_X86",
                "SCMP_ARCH_X32"
            ]
        }
    ],
    "syscalls": [
        {
            "names": [
                "accept", "accept4", "access", "adjtimex", "alarm", "bind", "brk", "capget", "capset",
                "chdir", "chmod", "chown", "chown32", "clock_getres", "clock_gettime", "clock_nanosleep",
                "close", "connect", "copy_file_range", "creat", "dup", "dup2", "dup3", "epoll_create",
                "epoll_create1", "epoll_ctl", "epoll_ctl_old", "epoll_pwait", "epoll_wait", "epoll_wait_old",
                "eventfd", "eventfd2", "execve", "execveat", "exit", "exit_group", "faccessat", "fadvise64",
                "fadvise64_64", "fallocate", "fanotify_mark", "fchdir", "fchmod", "fchmodat", "fchown",
                "fchown32", "fchownat", "fcntl", "fcntl64", "fdatasync", "fgetxattr", "flistxattr",
                "flock", "fork", "fremovexattr", "fsetxattr", "fstat", "fstat64", "fstatat64",
                "fstatfs", "fstatfs64", "fsync", "ftruncate", "ftruncate64", "futex", "getcpu",
                "getcwd", "getdents", "getdents64", "getegid", "getegid32", "geteuid", "geteuid32",
                "getgid", "getgid32", "getgroups", "getgroups32", "getitimer", "getpeername",
                "getpgid", "getpgrp", "getpid", "getppid", "getpriority", "getrandom", "getresgid",
                "getresgid32", "getresuid", "getresuid32", "getrlimit", "get_robust_list", "getrusage",
                "getsid", "getsockname", "getsockopt", "get_thread_area", "gettid", "gettimeofday",
                "getuid", "getuid32", "getxattr", "inotify_add_watch", "inotify_init", "inotify_init1",
                "inotify_rm_watch", "io_cancel", "ioctl", "io_destroy", "io_getevents", "ioprio_get",
                "ioprio_set", "io_setup", "io_submit", "ipc", "kill", "lchown", "lchown32", "lgetxattr",
                "link", "linkat", "listen", "listxattr", "llistxattr", "_llseek", "lremovexattr",
                "lseek", "lsetxattr", "lstat", "lstat64", "madvise", "memfd_create", "mincore",
                "mkdir", "mkdirat", "mknod", "mknodat", "mlock", "mlock2", "mlockall", "mmap",
                "mmap2", "mprotect", "mq_getsetattr", "mq_notify", "mq_open", "mq_timedreceive",
                "mq_timedsend", "mq_unlink", "mremap", "msgctl", "msgget", "msgrcv", "msgsnd",
                "msync", "munlock", "munlockall", "munmap", "nanosleep", "newfstatat", "_newselect",
                "open", "openat", "pause", "pipe", "pipe2", "poll", "ppoll", "prctl", "pread64",
                "preadv", "prlimit64", "pselect6", "pwrite64", "pwritev", "read", "readahead",
                "readlink", "readlinkat", "readv", "recv", "recvfrom", "recvmmsg", "recvmsg",
                "remap_file_pages", "removexattr", "rename", "renameat", "renameat2", "restart_syscall",
                "rmdir", "rt_sigaction", "rt_sigpending", "rt_sigprocmask", "rt_sigqueueinfo",
                "rt_sigreturn", "rt_sigsuspend", "rt_sigtimedwait", "rt_tgsigqueueinfo", "sched_getaffinity",
                "sched_getattr", "sched_getparam", "sched_get_priority_max", "sched_get_priority_min",
                "sched_getscheduler", "sched_setaffinity", "sched_setattr", "sched_setparam",
                "sched_setscheduler", "sched_yield", "seccomp", "select", "semctl", "semget",
                "semop", "semtimedop", "send", "sendfile", "sendfile64", "sendmmsg", "sendmsg",
                "sendto", "setfsgid", "setfsgid32", "setfsuid", "setfsuid32", "setgid", "setgid32",
                "setgroups", "setgroups32", "setitimer", "setpgid", "setpriority", "setregid",
                "setregid32", "setresgid", "setresgid32", "setresuid", "setresuid32", "setreuid",
                "setreuid32", "setrlimit", "set_robust_list", "setsid", "setsockopt", "set_thread_area",
                "set_tid_address", "setuid", "setuid32", "setxattr", "shmat", "shmctl", "shmdt",
                "shmget", "shutdown", "sigaltstack", "signalfd", "signalfd4", "sigreturn", "socket",
                "socketcall", "socketpair", "splice", "stat", "stat64", "statfs", "statfs64",
                "statx", "symlink", "symlinkat", "sync", "sync_file_range", "syncfs", "sysinfo",
                "tee", "tgkill", "time", "timer_create", "timer_delete", "timerfd_create",
                "timerfd_gettime", "timerfd_settime", "timer_getoverrun", "timer_gettime",
                "timer_settime", "times", "tkill", "truncate", "truncate64", "ugetrlimit",
                "umask", "uname", "unlink", "unlinkat", "utime", "utimensat", "utimes", "vfork",
                "vmsplice", "wait4", "waitid", "waitpid", "write", "writev"
            ],
            "action": "SCMP_ACT_ALLOW"
        }
    ]
}
EOF
    
    print_success "Security profiles created"
}

# Function to configure network security
configure_network_security() {
    print_status "Configuring network security..."
    
    # Create custom bridge network configuration
    cat <<EOF > /tmp/docker-compose.security.yml
version: '3.8'

networks:
  sql-synth-network:
    driver: bridge
    driver_opts:
      com.docker.network.bridge.enable_icc: "false"
      com.docker.network.bridge.enable_ip_masquerade: "true"
      com.docker.network.driver.mtu: "1500"
    ipam:
      config:
        - subnet: 172.20.0.0/16
          gateway: 172.20.0.1
EOF
    
    print_success "Network security configuration created"
}

# Function to set up monitoring and logging
setup_security_monitoring() {
    print_status "Setting up security monitoring..."
    
    # Create fail2ban configuration for Docker
    cat <<EOF | sudo tee /etc/fail2ban/jail.d/docker.conf > /dev/null
[docker-auth]
enabled = true
filter = docker-auth
logpath = /var/log/docker.log
maxretry = 3
bantime = 3600
findtime = 600
action = iptables-multiport[name=docker-auth, port="http,https"]

[docker-dos]
enabled = true
filter = docker-dos
logpath = /var/log/docker.log
maxretry = 10
bantime = 600
findtime = 60
action = iptables-multiport[name=docker-dos, port="http,https"]
EOF
    
    # Create audit rules for Docker
    cat <<EOF | sudo tee /etc/audit/rules.d/docker.rules > /dev/null
# Docker daemon audit rules
-w /usr/bin/docker -p rwxa -k docker
-w /var/lib/docker -p rwxa -k docker
-w /etc/docker -p rwxa -k docker
-w /lib/systemd/system/docker.service -p rwxa -k docker
-w /lib/systemd/system/docker.socket -p rwxa -k docker
-w /etc/default/docker -p rwxa -k docker
-w /etc/docker/daemon.json -p rwxa -k docker
-w /usr/bin/docker-containerd -p rwxa -k docker
-w /usr/bin/docker-runc -p rwxa -k docker
EOF
    
    print_success "Security monitoring configured"
}

# Function to create security validation script
create_security_validation() {
    print_status "Creating security validation script..."
    
    cat <<'EOF' > /tmp/docker-security-check.sh
#!/bin/bash
# Docker Security Validation Script

echo "🔒 Docker Security Check"
echo "========================"

# Check Docker daemon configuration
echo "\n📋 Docker Daemon Configuration:"
if [[ -f "/etc/docker/daemon.json" ]]; then
    echo "✅ daemon.json exists"
    if grep -q "userns-remap" /etc/docker/daemon.json; then
        echo "✅ User namespace remapping enabled"
    else
        echo "❌ User namespace remapping not enabled"
    fi
else
    echo "❌ daemon.json not found"
fi

# Check running containers security
echo "\n🐳 Container Security Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | while read line; do
    if [[ $line != *"NAMES"* ]]; then
        container_name=$(echo $line | awk '{print $1}')
        if [[ -n "$container_name" ]]; then
            # Check if container is running as root
            user=$(docker exec $container_name whoami 2>/dev/null || echo "unknown")
            if [[ "$user" == "root" ]]; then
                echo "⚠️  $container_name running as root"
            else
                echo "✅ $container_name running as $user"
            fi
        fi
    fi
done

# Check for privileged containers
echo "\n🔐 Privileged Container Check:"
privileged_containers=$(docker ps --filter "privileged=true" --format "{{.Names}}")
if [[ -z "$privileged_containers" ]]; then
    echo "✅ No privileged containers running"
else
    echo "❌ Privileged containers detected: $privileged_containers"
fi

# Check network security
echo "\n🌐 Network Security:"
docker network ls --format "table {{.Name}}\t{{.Driver}}\t{{.Scope}}"

# Check for exposed ports
echo "\n🚪 Port Exposure Check:"
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep -E "0\.0\.0\.0:|:::"
if [[ $? -eq 0 ]]; then
    echo "⚠️  Containers exposing ports to 0.0.0.0"
else
    echo "✅ No containers exposing ports to 0.0.0.0"
fi

# Check for read-only root filesystem
echo "\n📁 Read-only Filesystem Check:"
docker ps -q | xargs -I {} docker inspect {} --format "{{.Name}}: {{.HostConfig.ReadonlyRootfs}}" | grep false
if [[ $? -eq 0 ]]; then
    echo "⚠️  Some containers don't have read-only root filesystem"
else
    echo "✅ All containers have read-only root filesystem"
fi

echo "\n🏁 Security check completed"
EOF
    
    chmod +x /tmp/docker-security-check.sh
    print_success "Security validation script created at /tmp/docker-security-check.sh"
}

# Function to apply container runtime security
apply_runtime_security() {
    print_status "Applying container runtime security..."
    
    # Create docker-compose override with security settings
    cat <<EOF > docker-compose.security-override.yml
version: '3.8'

services:
  sql-synth-app:
    security_opt:
      - no-new-privileges:true
      - seccomp:./config/seccomp/sql-synth-profile.json
      - apparmor:docker-default
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
      - /var/run:noexec,nosuid,size=100m
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
      - CHOWN
      - SETGID
      - SETUID
    ulimits:
      nproc: 65535
      nofile:
        soft: 20000
        hard: 40000
        
  postgres:
    security_opt:
      - no-new-privileges:true
      - seccomp:unconfined
    read_only: false  # Database needs write access
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - DAC_OVERRIDE
      - FOWNER
      - SETGID
      - SETUID
        
  redis:
    security_opt:
      - no-new-privileges:true
      - seccomp:unconfined
    read_only: true
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
    cap_drop:
      - ALL
    cap_add:
      - SETGID
      - SETUID
EOF
    
    print_success "Security override configuration created"
}

# Function to run security benchmark
run_security_benchmark() {
    print_status "Running Docker security benchmark..."
    
    # Download and run Docker Bench Security if available
    if command -v docker-bench-security >/dev/null 2>&1; then
        docker-bench-security
    else
        print_warning "Docker Bench Security not installed. Consider installing it for comprehensive security assessment."
        print_status "You can install it with: git clone https://github.com/docker/docker-bench-security.git"
    fi
}

# Main execution
main() {
    print_status "Starting Docker security hardening..."
    
    check_root
    
    # Run hardening functions
    harden_docker_daemon
    setup_security_profiles
    configure_network_security
    setup_security_monitoring
    create_security_validation
    apply_runtime_security
    
    print_success "Docker security hardening completed!"
    print_status "Please restart Docker daemon to apply changes: sudo systemctl restart docker"
    print_status "Run the security validation: /tmp/docker-security-check.sh"
    
    # Optionally run security benchmark
    read -p "Run security benchmark now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_security_benchmark
    fi
}

# Run main function if script is executed directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi