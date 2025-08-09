## * NOTE: All the code in this file assumes
## that a permutation of length `n` contains
## values in the range [0, n).

import numpy as np


def inverse(p):
    s = np.empty_like(p)
    s[p] = np.arange(p.size)
    return s


def inversion(p: np.ndarray, inv: bool, gt: bool, left: bool):
    if inv:
        p = inverse(p)

    c = lambda x, y: x > y if gt else x < y

    s = np.empty_like(p)
    for i, e in enumerate(p):
        if left:
            s[i] = c(p[:i], p[i]).sum()
        else:
            s[i] = c(p[i:], p[i]).sum()
    return s


def rev_inversion(v: np.ndarray, inv: bool, gt: bool, left: bool):
    r = np.arange(v.size)

    if gt:
        r = r[::-1]

    mask = np.full_like(v, True, dtype=bool)
    s = np.empty_like(v)

    for i, e in enumerate(v if not left else v[::-1]):
        val = r[mask][e]
        s[i] = val
        mask[r == val] = False

    if left:
        s = s[::-1]

    if inv:
        s = inverse(s)
    return s

#
# Tests
# ----

# gt variants
# -----------

left_inversion_gt = lambda p: inversion(p, inv=False, gt=True, left=True)
rev_left_inversion_gt = lambda p: rev_inversion(p, inv=False, gt=True, left=True)

right_inversion_gt = lambda p: inversion(p, inv=False, gt=True, left=False)
rev_right_inversion_gt = lambda p: rev_inversion(p, inv=False, gt=True, left=False)

inversion_vector_gt = lambda p: inversion(p, inv=True, gt=True, left=False)
rev_inversion_vector_gt = lambda p: rev_inversion(p, inv=True, gt=True, left=False)

insertion_vector_gt = lambda p: inversion(p, inv=True, gt=True, left=True)
rev_insertion_vector_gt = lambda p: rev_inversion(p, inv=True, gt=True, left=True)


def random_permu(size):
    return np.random.permutation(size)


def is_permu(p):
    return (np.unique(p) == np.arange(p.size)).all()


def assert_equal(res, target):
    if not (res == target).all():
        print("\x1b[31;1mAssertion error! Result and target don't match:\x1b[0m")
        print(f"\t- Result: {res}")
        print(f"\t- Target: {target}")
        quit(1)


def test_left_inversion_gt():
    tests = [
        ([2, 1, 0, 4, 3], [0, 1, 2, 0, 1]),
        ([1, 4, 2, 3, 0], [0, 0, 1, 1, 4]),
        ([4, 1, 2, 3, 0], [0, 1, 1, 1, 4]),
    ]

    for permu, vec in tests:
        print("  *** Testing forward transformation")
        permu = np.array(permu)
        vec = np.array(vec)
        assert_equal(res=left_inversion_gt(permu), target=vec)

        # test the reverse function
        print("  *** Testing backward transformation")
        assert_equal(res=rev_left_inversion_gt(left_inversion_gt(permu)), target=permu)


def test_right_inversion_gt():
    tests = [
        ([2, 1, 0, 4, 3], [2, 2, 2, 0, 0]),
        ([1, 4, 2, 3, 0], [3, 0, 1, 0, 0]),
        ([4, 1, 3, 2, 0], [0, 2, 0, 0, 0]),
    ]

    for permu, vec in tests:
        print("  *** Testing forward transformation")
        permu = np.array(permu)
        vec = np.array(vec)
        assert_equal(res=right_inversion_gt(permu), target=vec)

        # test the reverse function
        print("  *** Testing backward transformation")
        assert_equal(res=rev_right_inversion_gt(right_inversion_gt(permu)), target=permu)


def test_inversion_insertion_vectors_gt(n=10, num_tests=10):
    for _ in range(num_tests):
        p = random_permu(n)

        # inversion vector
        inv = inversion_vector_gt(p)
        p_rev = rev_inversion_vector_gt(inv)
        assert_equal(res=p_rev, target=p_rev)

        # inversion vector
        ins = insertion_vector_gt(p)
        p_rev_2 = rev_insertion_vector_gt(ins)
        assert_equal(res=p_rev, target=p_rev_2)

# Lt variants
# -----------

left_inversion_lt = lambda p: inversion(p, inv=False, gt=False, left=True)
rev_left_inversion_lt = lambda p: rev_inversion(p, inv=False, gt=False, left=True)

right_inversion_lt = lambda p: inversion(p, inv=False, gt=False, left=False)
rev_right_inversion_lt = lambda p: rev_inversion(p, inv=False, gt=False, left=False)

inversion_vector_lt = lambda p: inversion(p, inv=True, gt=False, left=False)
rev_inversion_vector_lt = lambda p: rev_inversion(p, inv=True, gt=False, left=False)

insertion_vector_lt = lambda p: inversion(p, inv=True, gt=False, left=True)
rev_insertion_vector_lt = lambda p: rev_inversion(p, inv=True, gt=False, left=True)

def test_left_inversion_lt():
    tests = [
        ([2, 1, 0, 4, 3], [0, 0, 0, 3, 3]),
        ([1, 4, 2, 3, 0], [0, 1, 1, 2, 0]),
        ([4, 1, 2, 3, 0], [0, 0, 1, 2, 0]),
    ]

    for permu, vec in tests:
        print("  *** Testing forward transformation")
        permu = np.array(permu)
        vec = np.array(vec)
        assert_equal(res=left_inversion_lt(permu), target=vec)

        # test the reverse function
        print("  *** Testing backward transformation")
        assert_equal(res=rev_left_inversion_lt(left_inversion_lt(permu)), target=permu)


def test_right_inversion_lt():
    tests = [
        ([2, 1, 0, 4, 3], [2, 1, 0, 1, 0]),
        ([1, 4, 2, 3, 0], [1, 3, 1, 1, 0]),
        ([4, 1, 3, 2, 0], [4, 1, 2, 1, 0]),
    ]

    for permu, vec in tests:
        print("  *** Testing forward transformation")
        permu = np.array(permu)
        vec = np.array(vec)
        assert_equal(res=right_inversion_lt(permu), target=vec)

        # test the reverse function
        print("  *** Testing backward transformation")
        assert_equal(res=rev_right_inversion_lt(right_inversion_lt(permu)), target=permu)


def test_inversion_insertion_vectors_lt(n=10, num_tests=10):
    for _ in range(num_tests):
        p = random_permu(n)

        # inversion vector
        inv = inversion_vector_lt(p)
        p_rev = rev_inversion_vector_lt(inv)
        assert_equal(res=p_rev, target=p_rev)

        # inversion vector
        ins = insertion_vector_lt(p)
        p_rev_2 = rev_insertion_vector_lt(ins)
        assert_equal(res=p_rev, target=p_rev_2)


if __name__ == "__main__":
    print("[*] Running tests...\n")

    tests = [test_left_inversion_gt, test_right_inversion_gt, test_inversion_insertion_vectors_gt,
             test_left_inversion_lt, test_right_inversion_lt, test_inversion_insertion_vectors_lt]

    for test in tests:
        print(f"[**] {test.__name__}")
        test()
        print(f"\x1b[32;1mOK\x1b[0m")
