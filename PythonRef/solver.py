import numpy as np
from scipy.linalg import norm
import scipy.sparse as sparse
import scipy.sparse.linalg as sla
import spike as spike
import analysis as analysis
import matplotlib.pylab as plt
from timeit import default_timer as timer


def adjust_dimensions(dim, partitions):
    """
    Ajusta las dimensiones de la matriz al numero de particiones que buscamos
    dim: matrix dimension
    partitions: numero de particiones
    """
    if dim % partitions != 0:
        print 'Modifiying matrix dimensions!'
        blockdim = np.floor(dim / partitions)
        dim = blockdim * partitions

    return (dim)


def create_banded_matrix(N, offsets):
    '''
    Creates an sparse matrix with a number of diagonals given by offset array
    '''

    print 'Creating sparse matrix from offset vector', offsets
    # count total number of nnz in the Matrix
    nnz = 0
    for off in offsets:
        nnz += N - off

    # create coefficients
    coefficients = np.random.rand( len(offsets) * N).reshape(len(offsets),N)

    # create sparse matrix.
    Matrix = sparse.dia_matrix((coefficients, offsets), shape=(N, N))

    return Matrix.tocsc()


def compute_bandwidth(A, show_locations=False):
    """
    Computes the upper and lower bandwidth of the sparse matrix A
    """
    A = A.tocoo()

    # compute the bandwidth
    upper = abs(min(A.row - A.col))
    lower = abs(min(A.col - A.row))

    print 'Upper and lower bandwidth of A are (%d,%d)' % (upper, lower)

    if show_locations is True:
        # compute the position
        iupper = (A.row - A.col).argmin()
        ilower = (A.col - A.row).argmin()

        print 'Upper bound location (row %d, col %d)' % (A.row[iupper], A.col[iupper])
        print 'Lower bound location (row %d, col %d)' % (A.row[ilower], A.col[ilower])

    return (upper, lower)


def performance_analysis(A):
    """
    Calcula algunas metricas para estimar el rendmiento del solver.
    """
    (upper, lower) = compute_bandwidth(A)
    (m, n) = A.shape

    partitions = np.arange(2, 10)
    memory = np.zeros_like(partitions)

    for i in range(len(partitions)):
        p = partitions[i]
        mdim = adjust_dimensions(m, p)
        blockdim = np.ceil(mdim/p)

        if (p == 2):
            memory[i] = blockdim * upper + blockdim*lower
        else:
            memory[i] = blockdim * upper * (p-1) + blockdim * lower * (p-1)

    plt.plot(partitions, memory)
    plt.show()

def create_pentadiagonal(n):
    '''
    Crea un sistema tridiagonal de dimension N
    '''
    np.random.seed(314) # try to make it reproducible


    A = sparse.diags(np.random.rand(n) +1.0, 0) + \
        sparse.diags(np.random.rand(n-1), 1) + \
        sparse.diags(np.random.rand(n-1),-1) + \
        sparse.diags(np.random.rand(n-2), 2) + \
        sparse.diags(np.random.rand(n-2),-2)

    A = A.tocsr()

    # create rhs
    # B  = np.ones(shape=(n, nrhs), dtype=np.float64)

    return (A)

if __name__ == '__main__':
    print 'SPIKE solver'
    np.set_printoptions(precision=4, threshold=100, linewidth=120)

    # generamos la semilla del generador de numeros aleatorios
    np.random.seed(314)

    # numero de particiones empleadas en el primer nivel
    p = 2
    N = adjust_dimensions(10, p) #100000

    #A = create_banded_matrix(N, [0, 90, 50, -40, -90])
    create_pentadiagonal(N)

    ku, kl = compute_bandwidth(A.copy())
    nrhs = 1

    print 'dim(A) %d, (ku,kl) = (%d,%d) processes %d' % (A.shape[0], ku, kl, p)

    # b = np.random.rand(N*nrhs).reshape(N, nrhs)
    b = np.ones(N).reshape(N, nrhs)
    b_superlu = b.copy()
    b_spike = b.copy()


    # analysis.performance_analysis(A, b[:,0:2], 'analysis.rhs.1', pmin=1, pmax=50)
    # analysis.performance_analysis(A, b[:,0:45], 'analysis.rhs.45', pmin=1, pmax=50)
    # analysis.performance_analysis(A, b[:,0:90], 'analysis.rhs.90', pmin=1, pmax=50)
    # analysis.performance_analysis(A, b[:,0:180], 'analysis.rhs.180', pmin=1, pmax=50)
    # analysis.performance_analysis(A, b[:,0:360], 'analysis.rhs.360', pmin=1, pmax=50)
    # analysis.performance_analysis(A, b[:,0:720], 'analysis.rhs.720', pmin=1, pmax=50)
    # exit()


    just_spike = True

    if just_spike is True:
        print 'Using fully-implict SPIKE solver now...'
        start_t = timer()
        x_fspike, fspike_nnz = spike.spike_implicit_vw( A.copy(), b_spike, ku, kl, p, fully_implicit=True)
        end_t = timer()

        fspike_t = end_t - start_t

        print 'Time for iSPIKE %.5fs' % (end_t - start_t)
    else:
        print 'Computing reference solution'
        start_t = timer()
        LU = sla.splu(A.copy())
        superlu_nnz = LU.nnz - N
        x_superlu = LU.solve(b_superlu)
        end_t = timer()

        superlu_t = end_t - start_t

        print 'Tiempo superLU %.5f' % (superlu_t)

        print 'Using SPIKE solver now...'
        start_t = timer()
        x_spike, spike_nnz = spike.spike_naive(A.copy(), b_spike, ku, kl, p)
        end_t = timer()

        spike_t = end_t - start_t

        print 'Using fully-implict SPIKE solver now...'
        start_t = timer()
        x_fspike, fspike_nnz = spike.spike_implicit_vw( A.copy(), b_spike, ku, kl, p, fully_implicit=True)
        end_t = timer()

        fspike_t = end_t - start_t

        print 'Using semi-implict SPIKE solver now...'
        start_t = timer()
        x_ispike, ispike_nnz = spike.spike_implicit_vw(A.copy(), b_spike, ku, kl, p, fully_implicit=True)
        end_t = timer()

        ispike_t = end_t - start_t


        # print 'SuperLU', norm(b - A.dot(x_superlu))
        # print 'Error Spike', norm(b - A.dot(x_spike))

        print 'Tiempo SuperLU %.5fs, tiempo SPIKE %.5fs, tiempo para iSPIKE %.5fs, tiempo para fSPIKE %.5fs' % (superlu_t, spike_t, ispike_t, fspike_t)
        print 'Speed ups with respect to SuperLU:'
        print '    SPIKE              %.2fx' % ( superlu_t / spike_t)
        print '    semi-implict SPIKE %.2fx' % ( superlu_t / ispike_t)
        print '    fully-implict      %.2fx' % ( superlu_t / fspike_t)
        print 'nnz in SuperLU %d, nnz in SPIKE solver %d, nnz in iSPIKE solver %d' % (superlu_nnz, spike_nnz, ispike_nnz)

        for i in range( min(nrhs,10)):
            print '%.2d-th RHS - SPIKE error: %.2E iSPIKE error %.2E' %(i, norm(b[:, i] - A.dot(x_spike[:, i])), norm(b[:, i] - A.dot(x_ispike[:, i])))
