import numpy as np
# Use this for effective implementation of L-BFGS
from collections import defaultdict, deque
from utils import get_line_search_tool
import time


def conjugate_gradients(matvec, b, x_0, tolerance=1e-4, max_iter=None, trace=False, display=False):
    """
    Solves system Ax=b using Conjugate Gradients method.

    Parameters
    ----------
    matvec : function
        Implement matrix-vector product of matrix A and arbitrary vector x
    b : 1-dimensional np.array
        Vector b for the system.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
        Stop optimization procedure and return x_k when:
         ||Ax_k - b||_2 <= tolerance * ||b||_2
    max_iter : int, or None
        Maximum number of iterations. if max_iter=None, set max_iter to n, where n is
        the dimension of the space
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display:  bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['residual_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    if max_iter is None:
        max_iter = 100000
    history = defaultdict(list) if trace else None

    x_k = np.copy(x_0)
    r = b - matvec(x_0)
    p = r
    start_time = time.time()
    b_norm = np.linalg.norm(b)
    for _ in range(max_iter):
        if trace:
            history['time'].append(time.time() - start_time)
            history['residual_norm'].append(np.linalg.norm(r))
            history['x'].append(x_k) if len(x_0) <= 2 else history['x']

        alpha = r@r/(p@(matvec(p)))
        x_k = x_k + alpha*p
        r_old = r
        r = r - alpha*matvec(p)
        if np.linalg.norm(r) < tolerance * b_norm:
            if trace:
                history['time'].append(time.time() - start_time)
                history['residual_norm'].append(np.linalg.norm(r))
                history['x'].append(x_k) if len(x_0) <= 2 else history['x']
            return x_k, 'success', history

        beta = r@r/(r_old@r_old)
        p = r + beta*p

    return x_k, 'iterations_exceeded', history


def lbfgs(oracle, x_0, tolerance=1e-4, max_iter=500, memory_size=10,
          line_search_options=None, display=False, trace=False):
    """
    Limited-memory Broyden–Fletcher–Goldfarb–Shanno's method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func() and .grad() methods implemented for computing
        function value and its gradient respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    memory_size : int
        The length of directions history in L-BFGS method.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    start_time = time.time()
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    H = deque(maxlen=memory_size)

    for i in range(max_iter):
        grad_k = oracle.grad(x_k)
        grad_norm = np.linalg.norm(grad_k)
        if grad_norm < tolerance:
            if trace:
                history['time'].append(time.time() - start_time)
                history['grad_norm'].append(grad_norm)
                history['x'].append(x_k) if len(x_0) <= 2 else history['x']
                history['func'].append(oracle.func(x_k))
            return x_k, 'success', history

        d_k = -grad_k if i == 0 else Hgrad(H, grad_k)
        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        if trace:
            history['time'].append(time.time() - start_time)
            history['grad_norm'].append(grad_norm)
            history['x'].append(x_k) if len(x_0) <= 2 else history['x']
            history['func'].append(oracle.func(x_k))
        x_old = x_k
        grad_old = grad_k
        x_k = x_k + alpha*d_k
        grad_k = oracle.grad(x_k)
        H.append((x_k - x_old, grad_k - grad_old))

    return x_k, 'iterations_exceeded', history


def Hgrad(H: deque, grad_k):
    p, q = H[-1]
    gamma = p @ q / (q @ q)

    def recursion(v, depth):
        if depth == len(H):
            return gamma * v
        p, q = H.pop()
        H.appendleft((p, q))
        p_dot_v = p @ v
        p_dot_q = p @ q
        v = v - p_dot_v / p_dot_q * q
        z = recursion(v, depth+1)
        return z + (p_dot_v - q @ z) / p_dot_q * p
    return recursion(-grad_k, 0)


def hessian_free_newton(oracle, x_0, tolerance=1e-4, max_iter=500,
                        line_search_options=None, display=False, trace=False):
    """
    Hessian Free method for optimization.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess_vec() methods implemented for computing
        function value, its gradient and matrix product of the Hessian times vector respectively.
    x_0 : 1-dimensional np.array
        Starting point of the algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format is up to a student and is not checked in any way.
    trace:  bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
              the stopping criterion.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    d_k = - oracle.grad(x_0)
    start_time = time.time()
    for _ in range(max_iter):
        grad_k = oracle.grad(x_k)
        grad_norm = np.linalg.norm(grad_k)
        if grad_norm < tolerance:
            if trace:
                history['time'].append(time.time() - start_time)
                history['grad_norm'].append(grad_norm)
                history['x'].append(x_k) if len(x_0) <= 2 else history['x']
                history['func'].append(oracle.func(x_k))
            return x_k, 'success', history

        if trace:
            history['time'].append(time.time() - start_time)
            history['grad_norm'].append(grad_norm)
            history['x'].append(x_k) if len(x_0) <= 2 else history['x']
            history['func'].append(oracle.func(x_k))

        hess_k = oracle.hess_fast_call(x_k)
        ita_k = min(0.5, grad_norm)
        d_k, _, _ = conjugate_gradients(hess_k, -grad_k, d_k, ita_k)
        while grad_k@d_k > 0:
            ita_k *= 0.1
            d_k, _, _ = conjugate_gradients(hess_k, -grad_k, d_k, ita_k)
        alpha = line_search_tool.line_search(oracle, x_k, d_k)
        x_k = x_k + alpha*d_k

    return x_k, 'iterations_exceeded', history


def gradient_descent(oracle, x_0, tolerance=1e-4, max_iter=10000,
                     line_search_options=None, trace=False, display=False):
    history = defaultdict(list) if trace else None
    start_time = time.time()
    line_search_tool = get_line_search_tool(line_search_options)
    norm_grad_0 = np.linalg.norm(oracle.grad(x_0))
    x_k = np.copy(x_0)
    history['func'].append(oracle.func(x_0))
    history['grad_norm'].append(norm_grad_0)
    history['time'].append(0)
    if x_k.size <= 2:
        history['x'].append(x_k.copy())
    for _ in range(max_iter):
        f_x_k = oracle.func(x_k)
        grad_f_x_k = oracle.grad(x_k)
        grad_norm = np.linalg.norm(grad_f_x_k)
        if grad_norm**2 < tolerance*(norm_grad_0**2):
            return x_k, 'success', history
        alpha = line_search_tool.line_search(oracle, x_k, -grad_f_x_k)
        x_k -= alpha * grad_f_x_k
        if trace:
            history['func'].append(f_x_k)
            history['grad_norm'].append(grad_norm)
            history['time'].append(
                (time.time()-start_time))
            if x_k.size <= 2:
                history['x'].append(np.copy(x_k))
    return x_k, 'iterations_exceeded', history
