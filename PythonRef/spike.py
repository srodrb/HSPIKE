import cProfile
from timeit import default_timer as timer
import scipy.sparse.linalg as sla
from numpy import zeros, empty, empty_like
from scipy.sparse import eye, isspmatrix_csc


def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func



def spike_naive(A, f, ku, kl, partitions):
    '''
    A: matriz sparse de entrada
    f: lado derecho del sistema
    ku: upper bandwidth de A
    lu: lower bandwidth de A
    '''
    import scipy.sparse.linalg as sla
    from numpy import zeros, empty
    from scipy.sparse import eye, isspmatrix_csc
    import matplotlib.pylab as plt

    # local variables
    total_nnz = 0

    # Convert matrix to CSC format
    if not isspmatrix_csc(A):
        A.tocsc()

    # get matrix dimensions
    (mdim, n) = A.shape

    assert mdim == n, 'Matrix A have to square. Cant do anything.'

    # determine number of RHS
    nrhs = f.ndim
    if f.ndim == 1:
        nrhs = 1
        mb = len(f)
    else:
        (mb, nrhs) = f.shape

    assert mb == mdim, 'dimension mismatch on number of rows.'

    print 'A dim %d number of rhs %d' % (mdim, nrhs)

    # calculamos las dimensiones de las particiones
    assert mdim % partitions == 0, 'Cant divide matrix correctly'

    blockdim = mdim / partitions
    subblockdim = (kl + ku)

    print 'Block dimension %d, subblockdim %d' % (blockdim, subblockdim)

    # determine data type
    dprec = type(A.data[0])

    # create reduced system
    R = eye(subblockdim * partitions, dtype=dprec).tolil()
    yr = zeros(shape=(subblockdim*partitions, nrhs), dtype=dprec)
    y = zeros(shape=(mdim, nrhs), dtype=dprec)
    x = zeros(shape=(mdim, nrhs), dtype=dprec)

    # -------------------------------------------- Factorization stage
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        print 'A[%d:%d] sub[%d,%d]' % (bs, be, rs, re)

        if p is 0:
            print 'Factorizing first block (p=%d)' % (p)

            # factorize diagonal block
            lu = sla.splu(A[bs:be, bs:be])

            # get the number of nnz
            total_nnz = total_nnz + (lu.nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = lu.solve(f[bs:be, :])

            # Resolvemos para B->V
            V = lu.solve(A[bs:be, be:be+ku].todense())

            # copy the spikes back
            A[bs:be, be:be+ku] = V.copy()

            # Ensamblamos V en el sistema reducido
            R[rs:rs+ku, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-kl:blockdim, :]

        elif (p == partitions - 1):
            print 'Factorizing last block (p=%d)' % (p)

            # factorize diagonal block
            lu = sla.splu(A[bs:be, bs:be])

            # get the number of nnz
            total_nnz = total_nnz + (lu.nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = lu.solve(f[bs:be, :])

            # Resolvemos para W->C
            W = lu.solve(A[bs:be, bs-kl:bs].todense())

            # copy the spikes back
            A[bs:be, bs-kl:bs] = W.copy()

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

        else:
            print 'Factorizing middle block (p=%d)' % (p)

            print 'Block dimensions', be - bs

            # factorize diagonal block
            lu = sla.splu(A[bs:be, bs:be])

            # get the number of nnz
            total_nnz = total_nnz + (lu.nnz - blockdim)

            V = lu.solve(A[bs:be, be:be+ku].todense())
            y[bs:be, :] = lu.solve(f[bs:be, :])
            W = lu.solve(A[bs:be, bs-kl:bs].todense())

            # copy the spikes back
            A[bs:be, be:be+ku] = V.copy()
            A[bs:be, bs-kl:bs] = W.copy()

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

            # Ensamblamos V en el sistema reducido
            R[rs:rs+kl, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-ku:blockdim, :]

        # Ensamblamos el rhs del sistema reducido
        yr[rs:rs+kl, :] = y[bs:bs+kl, :]
        yr[re-ku:re, :] = y[be-ku:be, :]

    # -------------------------------------------- Solve stage
    lu = sla.splu(R.tocsc())
    xred = lu.solve(yr)

    # -------------------------------------------- Back assembly stage
    #
    #  Si no ensamblamos primero el sistema, nos encontraremos
    #  que al recuperar la solucion el primer bloque no tendra
    #  disponible x2t.
    #  Podemos cambiar el algoritmo para adaptarlo a este hecho
    #  y tener solo un bloque. Puesto que estos bloques hay que
    #  moverlos por la red, seria ventajoso.
    #
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        # first, we map the solution of the red system into the big one
        x[bs:bs+ku, :] = xred[rs:rs+ku, :]
        x[be-kl:be, :] = xred[re-kl:re, :]

    # -------------------------------------------- Retrieve solution stage
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        if p is 0:
            x[bs+ku:be-kl, :] = y[bs+ku:be-kl, :] - A[bs+ku:be-kl, be:be+ku].dot(x[be:be+ku, :])
        elif p is (partitions - 1):
            x[bs+ku:be-kl, :] = (y[bs+ku:be-kl, :] - A[bs+ku:be-kl, bs-kl:bs].dot(x[bs-kl:bs, :]))
        else:
            x[bs+ku:be-kl, :] = y[bs+ku:be-kl, :] - (A[bs+ku:be-kl, be:be+ku].dot(x[be:be+ku, :]) + A[bs+ku:be-kl, bs-kl:bs].dot(x[bs-kl:bs, :]))

    return (x, total_nnz)


def spike_implicit_vw(A, f, ku, kl, partitions, fully_implicit=True):
    '''
    Esta version del SPIKE emplea un uso implicito de las matrices Vi, Wi.

    A: matriz sparse de entrada
    f: lado derecho del sistema
    ku: upper bandwidth de A
    lu: lower bandwidth de A
    '''

    solver_start_t = timer()

    permc_spec = 'COLAMD'

    # local variables
    total_nnz = 0

    # Convert matrix to CSC format
    if not isspmatrix_csc(A):
        A.tocsc()

    # get matrix dimensions
    (mdim, n) = A.shape

    assert mdim == n, 'Matrix A have to square. Cant do anything.'

    # determine number of RHS
    nrhs = f.ndim
    if f.ndim == 1:
        nrhs = 1
        mb = len(f)
    else:
        (mb, nrhs) = f.shape

    assert mb == mdim, 'dimension mismatch on number of rows.'

    # calculamos las dimensiones de las particiones
    assert mdim % partitions == 0, 'Cant divide matrix correctly'

    blockdim = mdim / partitions
    subblockdim = (kl + ku)

    # determine data type
    dprec = type(A.data[0])

    # create reduced system
    R  = eye(subblockdim * partitions, dtype=dprec).tolil()
    yr = zeros(shape=(subblockdim*partitions, nrhs), dtype=dprec)
    y  = zeros(shape=(mdim, nrhs), dtype=dprec)
    x  = zeros(shape=(mdim, nrhs), dtype=dprec)
    Ai = []

    # -------------------------------------------- Factorization stage
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        if p is 0:
            # factorize diagonal block
            start_t = timer()
            Ai.append(sla.splu(A[bs:be, bs:be], permc_spec=permc_spec))
            end_t = timer()

            print 'Factorization time for the %d-th block %.5f' % (p, end_t - start_t)

            # get the number of nnz
            total_nnz = total_nnz + (Ai[p].nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = Ai[p].solve(f[bs:be, :])

            # Resolvemos para B->V
            V = Ai[p].solve(A[bs:be, be:be+ku].todense())

            # Ensamblamos V en el sistema reducido
            R[rs:rs+ku, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-kl:blockdim, :]

        elif (p == partitions - 1):
            # factorize diagonal block
            start_t = timer()
            Ai.append(sla.splu(A[bs:be, bs:be], permc_spec=permc_spec))
            end_t = timer()

            print 'Factorization time for the %d-th block %.5f' % (p, end_t - start_t)

            # get the number of nnz
            total_nnz = total_nnz + (Ai[p].nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = Ai[p].solve(f[bs:be, :])

            # Resolvemos para W->C
            W = Ai[p].solve(A[bs:be, bs-kl:bs].todense())

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

        else:
            # factorize diagonal block
            start_t = timer()
            Ai.append(sla.splu(A[bs:be, bs:be], permc_spec=permc_spec))
            end_t = timer()

            print 'Factorization time for the %d-th block %.5f' % (p, end_t - start_t)

            # get the number of nnz
            total_nnz = total_nnz + (Ai[p].nnz - blockdim)

            V = Ai[p].solve(A[bs:be, be:be+ku].todense())
            y[bs:be, :] = Ai[p].solve(f[bs:be, :])
            W = Ai[p].solve(A[bs:be, bs-kl:bs].todense())

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

            # Ensamblamos V en el sistema reducido
            R[rs:rs+kl, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-ku:blockdim, :]

        # Ensamblamos el rhs del sistema reducido
        yr[rs:rs+kl, :] = y[bs:bs+kl, :]
        yr[re-ku:re, :] = y[be-ku:be, :]

    # -------------------------------------------- Solve stage
    use_recursion = True

    R = R.tocsc()

    if use_recursion is True:
    	start_t = timer()
    	xred, innernnz = spike_core(R, yr, ku, kl, 8)
    	end_t = timer()
    	print 'Factorization and solve time for the reduced system %.5f using SPIKE core' % (end_t - start_t)
    else:
		start_t = timer()
		lu = sla.splu(R)
		xred = lu.solve(yr)
		end_t = timer()
		print 'Factorization and solve time for the reduced system %.5f' % (end_t - start_t)
	

    # -------------------------------------------- Back assembly stage
    #
    #  Si no ensamblamos primero el sistema, nos encontraremos
    #  que al recuperar la solucion el primer bloque no tendra
    #  disponible x2t.
    #  Podemos cambiar el algoritmo para adaptarlo a este hecho
    #  y tener solo un bloque. Puesto que estos bloques hay que
    #  moverlos por la red, seria ventajoso.
    #
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        # first, we map the solution of the red system into the big one
        x[bs:bs+ku, :] = xred[rs:rs+ku, :]
        x[be-kl:be, :] = xred[re-kl:re, :]

    # -------------------------------------------- Retrieve solution stage
    if fully_implicit is True:
        fl = empty_like(f[0:blockdim, :])

        for p in range(partitions):
            bs = p * blockdim  # big system starting point
            be = bs + blockdim  # big system ending point

            rs = p * subblockdim  # reduced system starting point
            re = rs + subblockdim  # reduced system ending point

            if p is 0:
                fl = f[bs:be, :].copy()
                fl[blockdim-kl:blockdim, :] -= A[be-kl:be, be:be+ku].dot(x[be:be+ku, :])

                x[bs+ku:be-kl, :] = Ai[p].solve(fl)[ku:blockdim-kl, :]
            elif p is (partitions - 1):
                fl = f[bs:be, :].copy()
                fl[0:ku, :] -= A[bs:bs+kl, bs-kl:bs].dot(x[bs-kl:bs, :])

                x[bs+ku:be-kl, :] = Ai[p].solve(fl)[ku:blockdim-kl, :]
            else:
                fl = f[bs:be, :].copy()
                fl[blockdim-kl:blockdim, :] -= A[be-kl:be, be:be+ku].dot(x[be:be+ku, :])
                fl[0:ku, :] -= A[bs:bs+kl, bs-kl:bs].dot(x[bs-kl:bs, :])

                x[bs+ku:be-kl, :] = Ai[p].solve(fl)[ku:blockdim-kl, :]
    else:
        for p in range(partitions):
            bs = p * blockdim  # big system starting point
            be = bs + blockdim  # big system ending point

            rs = p * subblockdim  # reduced system starting point
            re = rs + subblockdim  # reduced system ending point

            if p is 0:
                x[bs+ku:be-kl, :] = Ai[p].solve(f[bs:be, :] - A[bs:be, be:be+ku].dot(x[be:be+ku, :]))[ku:blockdim-kl, :]
            elif p is (partitions - 1):
                x[bs+ku:be-kl, :] = Ai[p].solve(f[bs:be, :] - A[bs:be, bs-kl:bs].dot(x[bs-kl:bs, :]))[ku:blockdim-kl, :]
            else:
                x[bs+ku:be-kl, :] = Ai[p].solve(f[bs:be, :] - A[bs:be, be:be+ku].dot(x[be:be+ku, :]) - A[bs:be, bs-kl:bs].dot(x[bs-kl:bs, :]))[ku:blockdim-kl, :]

    solver_end_t = timer()

    print 'SPIKE inner timer: %.5f seconds' % (solver_end_t - solver_start_t)


    return (x, total_nnz)


@profile
def spike_core(A, f, ku, kl, partitions, fully_implicit=True):
    '''
    A: matriz sparse de entrada
    f: lado derecho del sistema
    ku: upper bandwidth de A
    lu: lower bandwidth de A
    '''

    inner_solver_start_t = timer()

    permc_spec = 'COLAMD'

    # local variables
    total_nnz = 0

    # Convert matrix to CSC format
    if not isspmatrix_csc(A):
        A.tocsc()

    # get matrix dimensions
    (mdim, n) = A.shape

    assert mdim == n, 'Matrix A have to square. Cant do anything.'

    print 'Dimensiones del sistema ', A.shape

    # determine number of RHS
    nrhs = f.ndim
    if f.ndim == 1:
        nrhs = 1
        mb = len(f)
    else:
        (mb, nrhs) = f.shape

    assert mb == mdim, 'dimension mismatch on number of rows.'

    # calculamos las dimensiones de las particiones
    assert mdim % partitions == 0, 'Cant divide matrix correctly'

    blockdim = mdim / partitions
    subblockdim = (kl + ku)

    # determine data type
    dprec = type(A.data[0])

    # create reduced system
    R  = eye(subblockdim * partitions, dtype=dprec).tolil()
    yr = zeros(shape=(subblockdim*partitions, nrhs), dtype=dprec)
    y  = zeros(shape=(mdim, nrhs), dtype=dprec)
    x  = zeros(shape=(mdim, nrhs), dtype=dprec)
    Ai = []

    # -------------------------------------------- Factorization stage
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        if p is 0:
            # factorize diagonal block
            start_t = timer()
            Ai.append(sla.splu(A[bs:be, bs:be], permc_spec=permc_spec))
            end_t = timer()

            print 'Factorization time for the %d-th block %.5f' % (p, end_t - start_t)

            # get the number of nnz
            total_nnz = total_nnz + (Ai[p].nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = Ai[p].solve(f[bs:be, :])

            # Resolvemos para B->V
            V = Ai[p].solve(A[bs:be, be:be+ku].todense())

            # Ensamblamos V en el sistema reducido
            R[rs:rs+ku, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-kl:blockdim, :]

        elif (p == partitions - 1):
            # factorize diagonal block
            start_t = timer()
            Ai.append(sla.splu(A[bs:be, bs:be], permc_spec=permc_spec))
            end_t = timer()

            print 'Factorization time for the %d-th block %.5f' % (p, end_t - start_t)

            # get the number of nnz
            total_nnz = total_nnz + (Ai[p].nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = Ai[p].solve(f[bs:be, :])

            # Resolvemos para W->C
            W = Ai[p].solve(A[bs:be, bs-kl:bs].todense())

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

        else:
            # factorize diagonal block
            start_t = timer()
            Ai.append(sla.splu(A[bs:be, bs:be], permc_spec=permc_spec))
            end_t = timer()

            print 'Factorization time for the %d-th block %.5f' % (p, end_t - start_t)

            # get the number of nnz
            total_nnz = total_nnz + (Ai[p].nnz - blockdim)

            V = Ai[p].solve(A[bs:be, be:be+ku].todense())
            y[bs:be, :] = Ai[p].solve(f[bs:be, :])
            W = Ai[p].solve(A[bs:be, bs-kl:bs].todense())

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

            # Ensamblamos V en el sistema reducido
            R[rs:rs+kl, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-ku:blockdim, :]

        # Ensamblamos el rhs del sistema reducido
        yr[rs:rs+kl, :] = y[bs:bs+kl, :]
        yr[re-ku:re, :] = y[be-ku:be, :]

    # -------------------------------------------- Solve stage
    use_recursion = True


    if use_recursion is True:
    	start_t = timer()
    	xred, innernnz = spike_core_inner(R.tocsc(), yr, ku, kl, 4)
    	end_t = timer()
    	print 'Factorization and solve time for the reduced system %.5f using SPIKE core' % (end_t - start_t)
    else:
		start_t = timer()
		lu = sla.splu(R.tocsc())
		xred = lu.solve(yr)
		end_t = timer()
		print 'Factorization and solve time for the reduced system %.5f' % (end_t - start_t)
	

    # -------------------------------------------- Back assembly stage
    #
    #  Si no ensamblamos primero el sistema, nos encontraremos
    #  que al recuperar la solucion el primer bloque no tendra
    #  disponible x2t.
    #  Podemos cambiar el algoritmo para adaptarlo a este hecho
    #  y tener solo un bloque. Puesto que estos bloques hay que
    #  moverlos por la red, seria ventajoso.
    #
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        # first, we map the solution of the red system into the big one
        x[bs:bs+ku, :] = xred[rs:rs+ku, :]
        x[be-kl:be, :] = xred[re-kl:re, :]

    # -------------------------------------------- Retrieve solution stage
    if fully_implicit is True:
        fl = empty_like(f[0:blockdim, :])

        for p in range(partitions):
            bs = p * blockdim  # big system starting point
            be = bs + blockdim  # big system ending point

            rs = p * subblockdim  # reduced system starting point
            re = rs + subblockdim  # reduced system ending point

            if p is 0:
                fl = f[bs:be, :].copy()
                fl[blockdim-kl:blockdim, :] -= A[be-kl:be, be:be+ku].dot(x[be:be+ku, :])

                x[bs+ku:be-kl, :] = Ai[p].solve(fl)[ku:blockdim-kl, :]
            elif p is (partitions - 1):
                fl = f[bs:be, :].copy()
                fl[0:ku, :] -= A[bs:bs+kl, bs-kl:bs].dot(x[bs-kl:bs, :])

                x[bs+ku:be-kl, :] = Ai[p].solve(fl)[ku:blockdim-kl, :]
            else:
                fl = f[bs:be, :].copy()
                fl[blockdim-kl:blockdim, :] -= A[be-kl:be, be:be+ku].dot(x[be:be+ku, :])
                fl[0:ku, :] -= A[bs:bs+kl, bs-kl:bs].dot(x[bs-kl:bs, :])

                x[bs+ku:be-kl, :] = Ai[p].solve(fl)[ku:blockdim-kl, :]
    else:
        for p in range(partitions):
            bs = p * blockdim  # big system starting point
            be = bs + blockdim  # big system ending point

            rs = p * subblockdim  # reduced system starting point
            re = rs + subblockdim  # reduced system ending point

            if p is 0:
                x[bs+ku:be-kl, :] = Ai[p].solve(f[bs:be, :] - A[bs:be, be:be+ku].dot(x[be:be+ku, :]))[ku:blockdim-kl, :]
            elif p is (partitions - 1):
                x[bs+ku:be-kl, :] = Ai[p].solve(f[bs:be, :] - A[bs:be, bs-kl:bs].dot(x[bs-kl:bs, :]))[ku:blockdim-kl, :]
            else:
                x[bs+ku:be-kl, :] = Ai[p].solve(f[bs:be, :] - A[bs:be, be:be+ku].dot(x[be:be+ku, :]) - A[bs:be, bs-kl:bs].dot(x[bs-kl:bs, :]))[ku:blockdim-kl, :]

    inner_solver_end_t = timer()

    print 'Inner solver time %.5f seconds' % (inner_solver_end_t - inner_solver_start_t)

    return (x, total_nnz)

def spike_core_inner(A, f, ku, kl, partitions, fully_implicit=False):
    '''
    A: matriz sparse de entrada
    f: lado derecho del sistema
    ku: upper bandwidth de A
    lu: lower bandwidth de A
    '''

    permc_spec = 'COLAMD'

    # local variables
    total_nnz = 0

    # Convert matrix to CSC format
    if not isspmatrix_csc(A):
        A.tocsc()

    # get matrix dimensions
    (mdim, n) = A.shape

    assert mdim == n, 'Matrix A have to square. Cant do anything.'

    print 'Dimensiones del sistema ', A.shape

    # determine number of RHS
    nrhs = f.ndim
    if f.ndim == 1:
        nrhs = 1
        mb = len(f)
    else:
        (mb, nrhs) = f.shape

    assert mb == mdim, 'dimension mismatch on number of rows.'

    # calculamos las dimensiones de las particiones
    assert mdim % partitions == 0, 'Cant divide matrix correctly'

    blockdim = mdim / partitions
    subblockdim = (kl + ku)

    # determine data type
    dprec = type(A.data[0])

    # create reduced system
    R  = eye(subblockdim * partitions, dtype=dprec).tolil()
    yr = zeros(shape=(subblockdim*partitions, nrhs), dtype=dprec)
    y  = zeros(shape=(mdim, nrhs), dtype=dprec)
    x  = zeros(shape=(mdim, nrhs), dtype=dprec)
    Ai = []

    # -------------------------------------------- Factorization stage
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        if p is 0:
            # factorize diagonal block
            start_t = timer()
            Ai.append(sla.splu(A[bs:be, bs:be], permc_spec=permc_spec))
            end_t = timer()

            print 'Factorization time for the %d-th block %.5f' % (p, end_t - start_t)

            # get the number of nnz
            total_nnz = total_nnz + (Ai[p].nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = Ai[p].solve(f[bs:be, :])

            # Resolvemos para B->V
            V = Ai[p].solve(A[bs:be, be:be+ku].todense())

            # Ensamblamos V en el sistema reducido
            R[rs:rs+ku, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-kl:blockdim, :]

        elif (p == partitions - 1):
            # factorize diagonal block
            start_t = timer()
            Ai.append(sla.splu(A[bs:be, bs:be], permc_spec=permc_spec))
            end_t = timer()

            print 'Factorization time for the %d-th block %.5f' % (p, end_t - start_t)

            # get the number of nnz
            total_nnz = total_nnz + (Ai[p].nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = Ai[p].solve(f[bs:be, :])

            # Resolvemos para W->C
            W = Ai[p].solve(A[bs:be, bs-kl:bs].todense())

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

        else:
            # factorize diagonal block
            start_t = timer()
            Ai.append(sla.splu(A[bs:be, bs:be], permc_spec=permc_spec))
            end_t = timer()

            print 'Factorization time for the %d-th block %.5f' % (p, end_t - start_t)

            # get the number of nnz
            total_nnz = total_nnz + (Ai[p].nnz - blockdim)

            V = Ai[p].solve(A[bs:be, be:be+ku].todense())
            y[bs:be, :] = Ai[p].solve(f[bs:be, :])
            W = Ai[p].solve(A[bs:be, bs-kl:bs].todense())

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

            # Ensamblamos V en el sistema reducido
            R[rs:rs+kl, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-ku:blockdim, :]

        # Ensamblamos el rhs del sistema reducido
        yr[rs:rs+kl, :] = y[bs:bs+kl, :]
        yr[re-ku:re, :] = y[be-ku:be, :]

    # -------------------------------------------- Solve stage
    use_recursion = False


    if use_recursion is True:
    	start_t = timer()
    	xred, innernnz = spike_core(R.tocsc(), yr, ku, kl, 2)
    	end_t = timer()
    	print 'Factorization and solve time for the reduced system %.5f using SPIKE core' % (end_t - start_t)
    else:
		start_t = timer()
		lu = sla.splu(R.tocsc())
		xred = lu.solve(yr)
		end_t = timer()
		print 'Factorization and solve time for the reduced system %.5f' % (end_t - start_t)
	

    # -------------------------------------------- Back assembly stage
    #
    #  Si no ensamblamos primero el sistema, nos encontraremos
    #  que al recuperar la solucion el primer bloque no tendra
    #  disponible x2t.
    #  Podemos cambiar el algoritmo para adaptarlo a este hecho
    #  y tener solo un bloque. Puesto que estos bloques hay que
    #  moverlos por la red, seria ventajoso.
    #
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        # first, we map the solution of the red system into the big one
        x[bs:bs+ku, :] = xred[rs:rs+ku, :]
        x[be-kl:be, :] = xred[re-kl:re, :]

    # -------------------------------------------- Retrieve solution stage
    if fully_implicit is True:
        fl = empty_like(f[0:blockdim, :])

        for p in range(partitions):
            bs = p * blockdim  # big system starting point
            be = bs + blockdim  # big system ending point

            rs = p * subblockdim  # reduced system starting point
            re = rs + subblockdim  # reduced system ending point

            if p is 0:
                fl = f[bs:be, :].copy()
                fl[blockdim-kl:blockdim, :] -= A[be-kl:be, be:be+ku].dot(x[be:be+ku, :])

                x[bs+ku:be-kl, :] = Ai[p].solve(fl)[ku:blockdim-kl, :]
            elif p is (partitions - 1):
                fl = f[bs:be, :].copy()
                fl[0:ku, :] -= A[bs:bs+kl, bs-kl:bs].dot(x[bs-kl:bs, :])

                x[bs+ku:be-kl, :] = Ai[p].solve(fl)[ku:blockdim-kl, :]
            else:
                fl = f[bs:be, :].copy()
                fl[blockdim-kl:blockdim, :] -= A[be-kl:be, be:be+ku].dot(x[be:be+ku, :])
                fl[0:ku, :] -= A[bs:bs+kl, bs-kl:bs].dot(x[bs-kl:bs, :])

                x[bs+ku:be-kl, :] = Ai[p].solve(fl)[ku:blockdim-kl, :]
    else:
        for p in range(partitions):
            bs = p * blockdim  # big system starting point
            be = bs + blockdim  # big system ending point

            rs = p * subblockdim  # reduced system starting point
            re = rs + subblockdim  # reduced system ending point

            if p is 0:
                x[bs+ku:be-kl, :] = Ai[p].solve(f[bs:be, :] - A[bs:be, be:be+ku].dot(x[be:be+ku, :]))[ku:blockdim-kl, :]
            elif p is (partitions - 1):
                x[bs+ku:be-kl, :] = Ai[p].solve(f[bs:be, :] - A[bs:be, bs-kl:bs].dot(x[bs-kl:bs, :]))[ku:blockdim-kl, :]
            else:
                x[bs+ku:be-kl, :] = Ai[p].solve(f[bs:be, :] - A[bs:be, be:be+ku].dot(x[be:be+ku, :]) - A[bs:be, bs-kl:bs].dot(x[bs-kl:bs, :]))[ku:blockdim-kl, :]

    return (x, total_nnz)

def spike_recursive(A, f, ku, kl, partitions, use_recursion=False):
    '''
    A: matriz sparse de entrada
    f: lado derecho del sistema
    ku: upper bandwidth de A
    lu: lower bandwidth de A
    '''
    import scipy.sparse.linalg as sla
    from numpy import zeros, empty
    import scipy.sparse as sparse
    import matplotlib.pylab as plt

    # local variables
    total_nnz = 0

    # Convert matrix to CSC format
    if not sparse.isspmatrix_csc(A):
        A.tocsc()

    # get matrix dimensions
    (mdim, n) = A.shape

    assert mdim == n, 'Matrix A have to square. Cant do anything.'

    # determine number of RHS
    nrhs = f.ndim
    if f.ndim == 1:
        nrhs = 1
        mb = len(f)
    else:
        (mb, nrhs) = f.shape

    assert mb == mdim, 'dimension mismatch on number of rows.'

    print 'A dim %d number of rhs %d' % (mdim, nrhs)

    # calculamos las dimensiones de las particiones
    assert mdim % partitions == 0, 'Cant divide matrix correctly'

    blockdim = mdim / partitions
    subblockdim = (kl + ku)

    print 'Block dimension %d, subblockdim %d' % (blockdim, subblockdim)

    # determine data type
    dprec = type(A.data[0])

    # create reduced system
    R  = sparse.eye(subblockdim * partitions, dtype=dprec).tolil()
    yr = zeros(shape=(subblockdim*partitions, nrhs), dtype=dprec)
    y = zeros(shape=(mdim, nrhs), dtype=dprec)
    x = zeros(shape=(mdim, nrhs), dtype=dprec)

    # -------------------------------------------- Factorization stage
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        print 'A[%d:%d] sub[%d,%d]' % (bs, be, rs, re)

        if p is 0:
            print 'Factorizing first block (p=%d)' % (p)

            # factorize diagonal block
            lu = sla.splu(A[bs:be, bs:be])

            # get the number of nnz
            total_nnz = total_nnz + (lu.nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = lu.solve(f[bs:be, :])

            # Resolvemos para B->V
            V = lu.solve(A[bs:be, be:be+ku].todense())

            # copy the spikes back
            A[bs:be, be:be+ku] = V.copy()

            # Ensamblamos V en el sistema reducido
            R[rs:rs+ku, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-kl:blockdim, :]

        elif (p == partitions - 1):
            print 'Factorizing last block (p=%d)' % (p)

            # factorize diagonal block
            lu = sla.splu(A[bs:be, bs:be])

            # get the number of nnz
            total_nnz = total_nnz + (lu.nnz - blockdim)

            # Resolvemos y copiamos para y
            y[bs:be, :] = lu.solve(f[bs:be, :])

            # Resolvemos para W->C
            W = lu.solve(A[bs:be, bs-kl:bs].todense())

            # copy the spikes back
            A[bs:be, bs-kl:bs] = W.copy()

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

        else:
            print 'Factorizing middle block (p=%d)' % (p)

            print 'Block dimensions', be - bs

            # factorize diagonal block
            lu = sla.splu(A[bs:be, bs:be])

            # get the number of nnz
            total_nnz = total_nnz + (lu.nnz - blockdim)

            V = lu.solve(A[bs:be, be:be+ku].todense())
            y[bs:be, :] = lu.solve(f[bs:be, :])
            W = lu.solve(A[bs:be, bs-kl:bs].todense())

            # copy the spikes back
            A[bs:be, be:be+ku] = V.copy()
            A[bs:be, bs-kl:bs] = W.copy()

            # Ensamblamos W en el sistema
            R[rs:rs+kl, rs-kl:rs] = W[0:kl, :]
            R[re-kl:re, rs-kl:rs] = W[blockdim-kl:blockdim, :]

            # Ensamblamos V en el sistema reducido
            R[rs:rs+kl, re:re+ku] = V[0:ku, :]
            R[re-kl:re, re:re+ku] = V[blockdim-ku:blockdim, :]

        # Ensamblamos el rhs del sistema reducido
        yr[rs:rs+kl, :] = y[bs:bs+kl, :]
        yr[re-ku:re, :] = y[be-ku:be, :]

    # -------------------------------------------- Solve stage
    # plt.spy(R); plt.show(); exit()

    if use_recursion is False:
        lu = sla.splu(R.tocsc())
        xred = lu.solve(yr)
    else:
        R = R.tocsc()
        inner_partitions = 2
        print 'He usado %d particiones en el primer nivel. Ahora voy a usar %d' % (partitions, inner_partitions)
        print 'Dimension de R %d, dimension teorica %d' % (R.shape[0], 2*(kl+ku)*partitions)
        xred, innernnz = spike_core(R, yr, ku, kl, 2)

    # print 'Inner bandwidth %d %d' % (innerku, innerkl)
    # xred, innernnz = spike_core(R, yr, innerku, innerkl, 2)



    # -------------------------------------------- Back assembly stage
    #
    #  Si no ensamblamos primero el sistema, nos encontraremos
    #  que al recuperar la solucion el primer bloque no tendra
    #  disponible x2t.
    #  Podemos cambiar el algoritmo para adaptarlo a este hecho
    #  y tener solo un bloque. Puesto que estos bloques hay que
    #  moverlos por la red, seria ventajoso.
    #
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        # first, we map the solution of the red system into the big one
        x[bs:bs+ku, :] = xred[rs:rs+ku, :]
        x[be-kl:be, :] = xred[re-kl:re, :]

    # -------------------------------------------- Retrieve solution stage
    for p in range(partitions):
        bs = p * blockdim  # big system starting point
        be = bs + blockdim  # big system ending point

        rs = p * subblockdim  # reduced system starting point
        re = rs + subblockdim  # reduced system ending point

        # print 'bs: %d be %d kl %d ku %d' % (bs, be, ku, kl)
        # print 'x dims: ', x.shape
        # print 'A dims', A.shape

        if p is 0:
            x[bs+ku:be-kl, :] = y[bs+ku:be-kl, :] - A[bs+ku:be-kl, be:be+ku].dot(x[be:be+ku, :])
        elif p is (partitions - 1):
            x[bs+ku:be-kl, :] = (y[bs+ku:be-kl, :] - A[bs+ku:be-kl, bs-kl:bs].dot(x[bs-kl:bs, :]))
        else:
            x[bs+ku:be-kl, :] = y[bs+ku:be-kl, :] - (A[bs+ku:be-kl, be:be+ku].dot(x[be:be+ku, :]) + A[bs+ku:be-kl, bs-kl:bs].dot(x[bs-kl:bs, :]))

    return (x, total_nnz)
