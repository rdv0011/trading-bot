import tracemalloc
import gc
import time
import psutil, os

def profile_memory_during_backtest(func, *args, top_n=10, label=None, **kwargs):
    """
    Run a function (e.g., a single backtest) while profiling memory allocations
    using tracemalloc snapshots before and after.

    Parameters
    ----------
    func : callable
        Function to profile (e.g. run_backtest).
    *args, **kwargs :
        Arguments to pass to the function.
    top_n : int, optional
        Number of top memory allocation sources to print (default=10).
    label : str, optional
        Optional label to print before the report.

    Returns
    -------
    result : any
        The return value of the profiled function.
    """

    # Start tracemalloc if not already active
    if not tracemalloc.is_tracing():
        tracemalloc.start()

    gc.collect()
    snapshot_before = tracemalloc.take_snapshot()
    start_time = time.time()

    try:
        result = func(*args, **kwargs)
    except Exception as e:
        result = None
        print(f"[MemoryProfiler] Exception during {label or func.__name__}: {e}")

    snapshot_after = tracemalloc.take_snapshot()
    end_time = time.time()

    diff = snapshot_after.compare_to(snapshot_before, "filename")
    total_alloc = sum(stat.size_diff for stat in diff)

    print("\n" + "=" * 60)
    print(f"🧠 Memory Profile Report for {label or func.__name__}")
    print(f"⏱  Duration: {end_time - start_time:.2f} sec")
    print(f"📈  Total new memory allocated: {total_alloc / (1024 * 1024):.2f} MB")
    print("-" * 60)
    print(f"Top {top_n} memory increases:")
    for stat in diff[:top_n]:
        sign = "+" if stat.size_diff >= 0 else "-"
        print(f"{sign}{stat.size_diff / 1024:.1f} KiB | {stat.count_diff:+d} | {stat.traceback}")
    print("=" * 60 + "\n")

    gc.collect()
    return result


def print_mem_usage(prefix=""):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / 1024 / 1024
    print(f"[MEM] {prefix} RSS: {mem_mb:.2f} MB")