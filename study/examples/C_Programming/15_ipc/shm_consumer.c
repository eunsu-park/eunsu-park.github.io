/**
 * Shared Memory Consumer
 *
 * Reads data from POSIX shared memory written by shm_producer.
 * Run shm_producer first, then this program.
 *
 * Build: make
 * Usage: ./shm_consumer
 *
 * Note: Link with -lrt -lpthread on Linux
 */

#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <semaphore.h>

#define SHM_NAME  "/study_shm"
#define SEM_READY "/study_sem_ready"
#define SEM_DONE  "/study_sem_done"

#define NUM_ITEMS 10

typedef struct {
    int    id;
    double value;
    char   label[64];
} item_t;

typedef struct {
    int    count;
    item_t items[NUM_ITEMS];
} shared_data_t;

int main(void) {
    /* Open semaphores */
    sem_t *sem_ready = sem_open(SEM_READY, 0);
    sem_t *sem_done  = sem_open(SEM_DONE,  0);

    if (sem_ready == SEM_FAILED || sem_done == SEM_FAILED) {
        fprintf(stderr, "Run shm_producer first!\n");
        exit(EXIT_FAILURE);
    }

    /* Wait for data to be ready */
    printf("Consumer: waiting for producer...\n");
    sem_wait(sem_ready);

    /* Open shared memory */
    int shm_fd = shm_open(SHM_NAME, O_RDONLY, 0666);
    if (shm_fd < 0) {
        perror("shm_open");
        exit(EXIT_FAILURE);
    }

    shared_data_t *shm = mmap(NULL, sizeof(shared_data_t),
                               PROT_READ, MAP_SHARED, shm_fd, 0);
    if (shm == MAP_FAILED) {
        perror("mmap");
        exit(EXIT_FAILURE);
    }

    /* Read and display data */
    printf("Consumer: reading %d items from shared memory\n", shm->count);

    double total = 0.0;
    for (int i = 0; i < shm->count && i < NUM_ITEMS; i++) {
        printf("  [%d] %s = %.2f\n",
               shm->items[i].id, shm->items[i].label,
               shm->items[i].value);
        total += shm->items[i].value;
    }
    printf("  Total: %.2f, Average: %.2f\n",
           total, total / shm->count);

    /* Signal producer that we're done */
    sem_post(sem_done);

    /* Cleanup */
    sem_close(sem_ready);
    sem_close(sem_done);
    sem_unlink(SEM_READY);
    sem_unlink(SEM_DONE);
    munmap(shm, sizeof(shared_data_t));
    close(shm_fd);
    shm_unlink(SHM_NAME);

    printf("Consumer: done, resources cleaned up\n");
    return 0;
}
