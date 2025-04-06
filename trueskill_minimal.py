# A heavily stripped-down TrueSkill implementation
# for free-for-all and 1v1 with no partial-play or multi-backend complexity.
# Derived from the original code by Heungsub Lee (BSD license).

import math

###############################################################################
# Basic mathematics utilities: normal distribution, cdf/pdf/ppf, small matrix.
###############################################################################

inf = float('inf')

def erfc(x):
    """Complementary error function (approximation)"""
    z = abs(x)
    t = 1.0 / (1.0 + z / 2.0)
    r = t * math.exp(
        -z * z - 1.26551223 +
        t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
            -0.18628806 + t * (0.27886807 + t * (
                -1.13520398 + t * (1.48851587 + t * (
                    -0.82215223 + t * 0.17087277
                )))
            )))
        )))
    return 2.0 - r if x < 0 else r

def cdf(x, mu=0.0, sigma=1.0):
    """Cumulative distribution function for the normal distribution."""
    return 0.5 * erfc(-(x - mu) / (sigma * math.sqrt(2.0)))

def pdf(x, mu=0.0, sigma=1.0):
    """Probability density function for the normal distribution."""
    coeff = 1.0 / (math.sqrt(2.0 * math.pi) * sigma)
    exponent = -((x - mu) ** 2) / (2.0 * sigma * sigma)
    return coeff * math.exp(exponent)

def _gen_ppf(erfc_func):
    # We invert the cdf via an erfc-based approach
    def erfcinv(y):
        # This is an approximation to erfc^(-1)
        # (We won't bother making it bulletproof for extreme arguments.)
        # Adapted from original code.
        if y >= 2.0:
            return -100.0
        if y <= 0.0:
            return 100.0
        zero_point = (y < 1.0)
        if not zero_point:
            y = 2.0 - y
        t = math.sqrt(-2.0 * math.log(y / 2.0))
        x = -0.70711 * (
            (2.30753 + t * 0.27061) /
            (1.0 + t * (0.99229 + t * 0.04481))
            - t
        )
        for _ in range(2):
            err = erfc_func(x) - y
            x += err / (1.12837916709551257 * math.exp(-(x ** 2)) - x * err)
        return x if zero_point else -x

    ei = erfcinv

    def ppf(x, mu=0.0, sigma=1.0):
        return mu - sigma * math.sqrt(2.0) * ei(2.0 * x)
    return ppf

ppf = _gen_ppf(erfc)

class Gaussian:
    """
    A model for a normal distribution using precision form:
      pi = 1 / sigma^2
      tau = pi * mu
    """
    __slots__ = ('pi', 'tau')

    def __init__(self, mu=None, sigma=None, pi=0.0, tau=0.0):
        if mu is not None:
            if sigma is None:
                raise TypeError("sigma argument is required if mu is given")
            if sigma == 0:
                raise ValueError("sigma^2 must be > 0")
            self.pi = 1.0 / (sigma * sigma)
            self.tau = self.pi * mu
        else:
            self.pi = pi
            self.tau = tau

    @property
    def mu(self):
        return 0.0 if self.pi == 0.0 else (self.tau / self.pi)

    @property
    def sigma(self):
        return inf if self.pi == 0.0 else math.sqrt(1.0 / self.pi)

    def __mul__(self, other):
        return Gaussian(pi=self.pi + other.pi, tau=self.tau + other.tau)

    def __truediv__(self, other):
        return Gaussian(pi=self.pi - other.pi, tau=self.tau - other.tau)

    def __repr__(self):
        return f"N(mu={self.mu:.3f}, sigma={self.sigma:.3f})"

###############################################################################
# Matrix class, used by the "match quality" calculation for multiple players.
###############################################################################

class Matrix(list):
    """Simple 2D matrix of floats, enough to handle the match-quality code."""
    def __init__(self, src, height=None, width=None):
        if isinstance(src, list):
            # We assume a rectangular numeric list of lists
            row_lengths = {len(r) for r in src}
            if len(row_lengths) != 1:
                raise ValueError("not a rectangular array")
            super().__init__(src)
        elif isinstance(src, dict):
            # build from dict {(r,c): val, ...}
            if height is None or width is None:
                max_r = max_c = 0
                for (r,c) in src.keys():
                    if r+1>max_r: max_r = r+1
                    if c+1>max_c: max_c = c+1
                if height is None:
                    height = max_r
                if width is None:
                    width = max_c
            newdata = []
            for r in range(height):
                row = []
                for c in range(width):
                    row.append(src.get((r,c),0.0))
                newdata.append(row)
            super().__init__(newdata)
        else:
            raise TypeError("Matrix src must be list-of-lists or dict")
        self._height = len(self)
        self._width = len(self[0])

    @property
    def height(self):
        return self._height

    @property
    def width(self):
        return self._width

    def __mul__(self, other):
        # standard matrix multiply
        if self.width != other.height:
            raise ValueError("bad size for matrix multiplication")
        outH, outW = self.height, other.width
        result = []
        for r in range(outH):
            row = []
            for c in range(outW):
                val = 0.0
                for i in range(self.width):
                    val += self[r][i]*other[i][c]
                row.append(val)
            result.append(row)
        return Matrix(result)

    def __rmul__(self, scalar):
        if not isinstance(scalar, (int,float)):
            raise TypeError("can only multiply by a number")
        newdata = []
        for row in self:
            newdata.append([scalar*val for val in row])
        return Matrix(newdata)

    def transpose(self):
        # produce M^T
        newdata = []
        for c in range(self.width):
            row = []
            for r in range(self.height):
                row.append(self[r][c])
            newdata.append(row)
        return Matrix(newdata)

    def determinant(self):
        # We only need a limited range for match quality, so let's do
        # a simple row-based approach. (Will be slow for large dimension.)
        if self.height != self.width:
            raise ValueError("not a square matrix")
        tmp = [row[:] for row in self]
        n = self.height
        det = 1.0
        for c in range(n-1,0,-1):
            # partial pivot for column c
            pivot_val = 0.0
            pivot_row = c
            for r in range(c+1):
                val = abs(tmp[r][c])
                if val>pivot_val:
                    pivot_val = val
                    pivot_row = r
            if pivot_val==0.0:
                return 0.0
            if pivot_row!=c:
                tmp[c], tmp[pivot_row] = tmp[pivot_row], tmp[c]
                det = -det
            det *= tmp[c][c]
            fact = -1.0/tmp[c][c]
            for rr in range(c):
                f = fact*tmp[rr][c]
                for cc in range(c+1):
                    tmp[rr][cc] += f*tmp[c][cc]
        det *= tmp[0][0]
        return det

    def inverse(self):
        # naive approach
        d = self.determinant()
        if d==0.0:
            raise ValueError("singular matrix")
        if self.height==1:
            return Matrix([[1.0/self[0][0]]])
        return (1.0/d)*self.adjugate()

    def adjugate(self):
        n = self.height
        if n!=self.width:
            raise ValueError("not square")
        if n==2:
            [[a,b],[c,d]] = self
            return Matrix([[ d, -b],[-c,  a]])
        # general
        def minor(mat, row, col):
            sub = []
            for rr in range(n):
                if rr==row: continue
                rowdat = []
                for cc in range(n):
                    if cc==col: continue
                    rowdat.append(mat[rr][cc])
                sub.append(rowdat)
            return Matrix(sub)
        out = {}
        for r in range(n):
            for c in range(n):
                sign = -1 if (r+c)%2 else 1
                out[r,c] = sign * minor(self, r,c).determinant()
        # then transpose
        # but we can fill that in transposed right away, i.e. out[c,r] = ...
        trans_out = {}
        for (r,c), val in out.items():
            trans_out[c,r] = val
        return Matrix(trans_out, n, n)

###############################################################################
# Factor graph nodes specialized for single-person "teams" in free-for-all.
###############################################################################

class Variable(Gaussian):
    """A Gaussian variable node in the factor graph."""
    def __init__(self):
        super().__init__(pi=0.0, tau=0.0)
        self.messages = {}

    def set_val(self, newval):
        # Returns the magnitude of the delta
        delta_pi = abs(self.pi - newval.pi)
        if delta_pi == inf:
            return 0.0
        delta = max(abs(self.tau - newval.tau), math.sqrt(delta_pi))
        self.pi, self.tau = newval.pi, newval.tau
        return delta

    def update_message(self, factor, new_msg):
        old_msg = self.messages.get(factor, Gaussian(pi=0.0, tau=0.0))
        # out with old, in with new
        self.messages[factor] = new_msg
        # adjust self by dividing out old_msg, multiplying in new_msg
        return self.set_val((self / old_msg) * new_msg)

    def update_value(self, factor, new_val):
        old_msg = self.messages[factor]
        # new message = new_val * old_msg / self
        self.messages[factor] = (new_val * old_msg) / self
        return self.set_val(new_val)

class Factor:
    """Base class for factors in the factor graph."""
    def __init__(self, variables):
        self.vars = variables
        for v in variables:
            v.messages[self] = Gaussian(pi=0.0, tau=0.0)

class PriorFactor(Factor):
    """Fuses with a prior rating."""
    def __init__(self, var, rating, dynamic):
        super().__init__([var])
        self.mu = rating.mu
        self.sigma = math.sqrt(rating.sigma**2 + dynamic**2)

    def down(self):
        # set the var to N(mu, sigma)
        val = Gaussian(mu=self.mu, sigma=self.sigma)
        return self.vars[0].update_value(self, val)

class LikelihoodFactor(Factor):
    """Relates rating to performance via Beta variance."""
    def __init__(self, rating_var, perf_var, beta_sq):
        super().__init__([rating_var, perf_var])
        self.rating_var = rating_var
        self.perf_var   = perf_var
        self.beta_sq    = beta_sq

    def _calc_a(self, v):
        return 1.0/(1.0 + self.beta_sq*v.pi)

    def down(self):
        # update performance from rating
        r = self.rating_var
        msg_r = r / r.messages[self]
        a = self._calc_a(msg_r)
        new_pi  = a * msg_r.pi
        new_tau = a * msg_r.tau
        return self.perf_var.update_message(self, Gaussian(pi=new_pi, tau=new_tau))

    def up(self):
        # update rating from performance
        p = self.perf_var
        msg_p = p / p.messages[self]
        a = self._calc_a(msg_p)
        return self.rating_var.update_message(
            self, Gaussian(pi=a*msg_p.pi, tau=a*msg_p.tau)
        )

class SumFactor(Factor):
    """
    In general TrueSkill, we’d sum multiple performance variables. But here,
    each “team” is exactly one rating, so we only need to do
        diff = perf_j - perf_i
    style usage. We'll do a minimal version of the sum factor here.
    """
    def __init__(self, sum_var, varA, varB, coeffA, coeffB):
        super().__init__([sum_var, varA, varB])
        self.sum_var = sum_var
        self.varA    = varA
        self.varB    = varB
        self.ca      = coeffA
        self.cb      = coeffB

    def down(self):
        # sum_var = ca*varA + cb*varB
        # new_msg for sum_var
        # We do this by removing old msgs from varA,varB, then combining
        oldA = self.varA.messages[self]
        oldB = self.varB.messages[self]
        newA = self.varA / oldA
        newB = self.varB / oldB

        pi_inv = 0.0
        mu = 0.0
        for (v, c) in [(newA, self.ca),(newB, self.cb)]:
            mu += c*v.mu
            if pi_inv==inf:
                continue
            try:
                pi_inv += (c*c)/v.pi
            except ZeroDivisionError:
                pi_inv = inf

        if pi_inv==0.0:
            # no changes
            return 0.0
        pi = 1.0/pi_inv
        tau = pi*mu
        return self.sum_var.update_message(self, Gaussian(pi=pi, tau=tau))

    def up(self, var_idx):
        # If var_idx==0 => updating varA
        # If var_idx==1 => updating varB
        # We'll assume (sum_var is index -1).
        if var_idx==0:
            # varA = (sum_var - cb*varB)/ca
            pass
        elif var_idx==1:
            # varB = (sum_var - ca*varA)/cb
            pass
        # But for the factor graph scheduling logic in TrueSkill,
        # we typically only do `sumFactor.down()` for the sum var,
        # and `sumFactor.up()` for one child at a time. We'll do a minimal approach.
        if var_idx==0:
            # varA from sum_var, varB
            oldSum = self.sum_var.messages[self]
            oldB   = self.varB.messages[self]
            newSum = self.sum_var / oldSum
            newB   = self.varB / oldB
            # do the standard formula:
            denom_pi_inv = 0.0
            denom_mu     = 0.0
            # sum_var = ca * varA + cb * varB => varA = (1/ca)*sum_var - (cb/ca)*varB
            cA_inv = 1.0/self.ca
            cB_div = -(self.cb/self.ca)
            for (v, c) in [(newSum, cA_inv),(newB, cB_div)]:
                denom_mu += c*v.mu
                try:
                    denom_pi_inv += (c*c)/v.pi
                except ZeroDivisionError:
                    denom_pi_inv=inf
            if denom_pi_inv==0.0:
                return 0.0
            pi = 1.0/denom_pi_inv
            tau= pi*denom_mu
            return self.varA.update_message(self, Gaussian(pi=pi, tau=tau))

        elif var_idx==1:
            # varB from sum_var, varA
            oldSum = self.sum_var.messages[self]
            oldA   = self.varA.messages[self]
            newSum = self.sum_var / oldSum
            newA   = self.varA / oldA
            cB_inv = 1.0/self.cb
            cA_div = -(self.ca/self.cb)
            denom_pi_inv=0.0
            denom_mu=0.0
            for (v, c) in [(newSum, cB_inv),(newA, cA_div)]:
                denom_mu += c*v.mu
                try:
                    denom_pi_inv += (c*c)/v.pi
                except ZeroDivisionError:
                    denom_pi_inv=inf
            if denom_pi_inv==0.0:
                return 0.0
            pi = 1.0/denom_pi_inv
            tau= pi*denom_mu
            return self.varB.update_message(self, Gaussian(pi=pi, tau=tau))
        return 0.0

class TruncateFactor(Factor):
    """Applies a win or draw truncation across the difference variable."""
    def __init__(self, diff_var, draw_margin, is_draw):
        super().__init__([diff_var])
        self.diff_var = diff_var
        self.draw_margin = draw_margin
        self.is_draw = is_draw

    def up(self):
        # We read the current variable (div = full distribution)
        # Then apply the appropriate v/w functions for the truncated side.
        val = self.diff_var
        msg = val.messages[self]
        div = val / msg
        sqrt_pi = math.sqrt(div.pi)

        # x = diff / sigma, margin = draw_margin / sigma
        x = div.tau / sqrt_pi
        m = self.draw_margin * sqrt_pi

        if self.is_draw:
            # vDraw
            # a= m - |x|, b= -m - |x|
            absx = abs(x)
            a = m - absx
            b = -m - absx
            denom = cdf(a) - cdf(b)
            if denom<=0.0:
                # floating error => skip
                return 0.0
            num = pdf(b) - pdf(a)
            v = (num/denom)*(-1.0 if x<0 else 1.0)
            w = (v*v) + (a*pdf(a) - b*pdf(b))/denom
        else:
            # vWin
            # x - margin => denominator = cdf(x - m)
            d = x - m
            denom = cdf(d)
            if denom<=0.0:
                return 0.0
            v = pdf(d)/denom
            w = v*(v + d)

        new_pi = div.pi / (1.0 - w)
        new_tau= (div.tau + sqrt_pi*v)/(1.0 - w)
        gf = Gaussian(pi=new_pi, tau=new_tau)
        return val.update_value(self, gf)

###############################################################################
# TrueSkill environment for 1v1 or N-player free-for-all, no partial weights.
###############################################################################

MU               = 25.0
SIGMA            = MU/3.0
BETA             = SIGMA/2.0
TAU              = SIGMA/100.0
DRAW_PROBABILITY = 0.1
DELTA            = 0.0001

def calc_draw_margin(draw_probability, num_players, beta):
    # For 2 teams with "num_players" total, the original code used:
    # margin = ppf((draw_probability+1)/2) * sqrt(num_players)*beta
    # In a free-for-all, we similarly treat the "size" as sum of players in
    # two "teams" for a difference. So for 1v1, it's 2 players total, etc.
    return ppf((draw_probability+1.0)/2.0)*math.sqrt(num_players)*beta

class Rating(Gaussian):
    """
    A player's skill as a Gaussian (mu, sigma).
    Default mu, sigma come from global_env() if not specified.
    """
    def __init__(self, mu=None, sigma=None):
        if mu is None:
            mu = global_env().mu
        if sigma is None:
            sigma = global_env().sigma
        super().__init__(mu=mu, sigma=sigma)

    def __repr__(self):
        return f"Rating(mu={self.mu:.3f}, sigma={self.sigma:.3f})"

class TrueSkill:
    """
    Simplified TrueSkill environment: single-person "teams",
    suitable for free-for-all or 1v1.
    """
    def __init__(self, mu=MU, sigma=SIGMA, beta=BETA, tau=TAU,
                 draw_probability=DRAW_PROBABILITY):
        self.mu = mu
        self.sigma = sigma
        self.beta  = beta
        self.tau   = tau
        self.draw_probability = draw_probability

    def create_rating(self, mu=None, sigma=None):
        if mu is None: mu = self.mu
        if sigma is None: sigma = self.sigma
        return Rating(mu=mu, sigma=sigma)

    def expose(self, rating):
        # Equivalent to rating.mu - k*rating.sigma, where k=mu/sigma
        # but we store the standard ratio = mu/sigma => rating.mu - (mu/sigma)*rating.sigma
        # Actually the official formula is rating.mu - 3*rating.sigma if default is used
        return rating.mu - 3.0*rating.sigma

    def rate(self, ratings, ranks=None, min_delta=DELTA):
        """
        ratings: list of N single-person Ratings.
        ranks:   list of the same length, giving each player's finish rank
                 (lower is better). If omitted, we assume ranks=[0,1,2,...].
        returns: new list of updated Ratings, same order.
        """
        n = len(ratings)
        if n<2:
            raise ValueError("Need at least 2 players for a match")
        if ranks is None:
            ranks = list(range(n))
        if len(ranks)!=n:
            raise ValueError("ranks must match number of players")

        # sort players by rank
        idx_and_ranks = list(enumerate(ranks))
        idx_and_ranks.sort(key=lambda x:x[1])
        sorted_indices = [p[0] for p in idx_and_ranks]
        sorted_ratings= [ratings[i] for i in sorted_indices]
        sorted_ranks  = [r for (_,r) in idx_and_ranks]

        # build factor graph
        rating_vars = [Variable() for _ in range(n)]
        perf_vars   = [Variable() for _ in range(n)]
        diff_vars   = [Variable() for _ in range(n-1)]  # consecutive diffs

        # initialize prior & performance-likelihood factors
        # for each player
        priors = []
        likes  = []
        for i in range(n):
            priors.append( PriorFactor(rating_vars[i], sorted_ratings[i], self.tau) )
            likes.append(  LikelihoodFactor(rating_vars[i], perf_vars[i], self.beta**2) )

        # sum factors for differences
        # diff[i] = perf[i+1] - perf[i]
        # then truncate factor depends on whether ranks[i+1] == ranks[i] (a tie) or not
        sum_factors    = []
        truncate_facts = []
        draw_margin = calc_draw_margin(self.draw_probability, n, self.beta)
        for i in range(n-1):
            sum_factors.append( SumFactor(diff_vars[i], perf_vars[i+1], perf_vars[i],
                                          +1.0, -1.0) )
            is_draw = (sorted_ranks[i]==sorted_ranks[i+1])
            t = TruncateFactor(diff_vars[i], draw_margin, is_draw)
            truncate_facts.append(t)

        # run the schedule
        # 1) Down through all prior/performance factors
        for f in priors: f.down()
        for f in likes:
            f.down()

        # 2) For the difference sum factors, we do a left->right pass and
        #    incorporate truncate factors, then right->left, repeating
        #    until changes are small or iteration limit.
        for _ in range(20):
            delta = 0.0
            # left->right
            for i in range(n-1):
                sum_factors[i].down()    # update diff
                d = truncate_facts[i].up()  # apply truncation
                delta = max(delta,d)
                sum_factors[i].up(1)    # up to perf[i+1]
            # right->left
            for i in range(n-2,-1,-1):
                sum_factors[i].down()    # diff
                d = truncate_facts[i].up()
                delta = max(delta,d)
                sum_factors[i].up(0)    # up to perf[i]

            if delta<min_delta:
                break

        # upward final for performance->rating
        for f in likes:
            f.up()

        # read out final rating variables
        # Then reorder to original
        new_ratings = []
        for i,rv in enumerate(rating_vars):
            # rating is rv's current distribution
            new_ratings.append(Rating(mu=rv.mu, sigma=rv.sigma))
        # unsort them
        unsorted = [None]*n
        for i in range(n):
            unsorted[sorted_indices[i]] = new_ratings[i]
        return unsorted

    def quality(self, ratings):
        """
        For free-for-all with N single-person "teams," uses the matrix approach
        from the original code, returning a "match quality" in [0..1] range.
        Interpreted as approximate draw probability for the entire match.
        """
        n = len(ratings)
        if n<2:
            return 1.0
        # Flatten everything
        mus    = [r.mu for r in ratings]
        sigsqs = [r.sigma*r.sigma for r in ratings]

        # mean vector
        from_dict = {}
        for i,m in enumerate(mus):
            from_dict[i,0] = m
        mean_vec = Matrix(from_dict, n, 1)

        # diag of variances
        var_dict = {}
        for i,v in enumerate(sigsqs):
            var_dict[i,i] = v
        var_mat = Matrix(var_dict, n, n)

        # Build “A” matrix that compares consecutive players
        # in rank order. For free-for-all “ranking,” we do (n-1) constraints.
        # But the official code does an arrangement of "teams." We'll do
        # the original approach: put them in a chain. We'll be naive:
        # we do N-1 rows, row i has 1 for i, -1 for i+1, 0 for others
        a_data = {}
        for i in range(n-1):
            a_data[i, i]   = +1.0
            a_data[i, i+1] = -1.0
        # This yields an (n-1) x n matrix
        a_mat = Matrix(a_data, height=(n-1), width=n)
        a_t   = a_mat.transpose()

        # (Beta^2)*A*A^T + A*Var*A^T
        # see original code. We'll adapt:
        beta2 = self.beta**2
        # We'll build:
        # _ATA = (beta^2)*(a_mat * a_t)
        # _AVAt= a_mat * var_mat * a_t
        # middle = _ATA + _AVAt
        # Then do a fancy exponent/determinant. See original.
        ATA  = (a_mat * a_t)
        for r in range(len(ATA)):
            for c in range(len(ATA[0])):
                ATA[r][c] *= beta2
        AVA  = a_mat * var_mat * a_t
        # sum them
        if (ATA.height!=AVA.height) or (ATA.width!=AVA.width):
            # shouldn't happen
            return 0.0

        # we do middle = ATA + AVA
        # then e_arg = -0.5 * mean^T * A^T * middle^-1 * A * mean
        # s_arg = det(ATA)/det(middle)
        # final = exp(e_arg)*sqrt(s_arg)
        middle_dict = {}
        for r in range(ATA.height):
            for c in range(ATA.width):
                middle_dict[r,c] = ATA[r][c] + AVA[r][c]
        middle = Matrix(middle_dict, ATA.height, ATA.width)
        try:
            mid_inv = middle.inverse()
        except ValueError:
            return 0.0  # singular => no quality

        # start = (mean^T * a_t)
        start = (mean_vec.transpose() * a_mat.transpose())
        # note a_mat.transpose() is dimension n x (n-1), but we must be careful
        # The original code used partial sums for "2-team" scenario. We'll adapt:
        # We'll do: start * mid_inv * (a_mat * mean_vec)
        # end = a_mat*mean_vec
        end   = a_mat*mean_vec
        sm    = start * mid_inv * end
        # it’s effectively a 1x1 matrix, so use [0][0]
        e_arg = -0.5*sm[0][0]

        # determinant ratio
        try:
            s_arg = ATA.determinant()/middle.determinant()
            if s_arg<=0.0:
                return 0.0
        except ValueError:
            return 0.0

        val = math.exp(e_arg)*math.sqrt(s_arg)
        if val>1.0:
            val=1.0
        if val<0.0:
            val=0.0
        return val

    def make_as_global(self):
        setup(env=self)
        return self

def global_env():
    """Get the globally configured TrueSkill environment."""
    # We'll stash it in a module variable:
    return _GLOBAL_ENV[0]

def setup(mu=MU, sigma=SIGMA, beta=BETA, tau=TAU,
          draw_probability=DRAW_PROBABILITY, env=None):
    """Set the global environment. If env is provided, use that."""
    if env is None:
        env = TrueSkill(mu, sigma, beta, tau, draw_probability)
    _GLOBAL_ENV[0] = env
    return env

_GLOBAL_ENV = [TrueSkill()]  # store a single item list

###############################################################################
# Top-level convenience: rate, quality, rate_1vs1, quality_1vs1, expose
###############################################################################

def rate(ratings, ranks=None, min_delta=DELTA):
    return global_env().rate(ratings, ranks, min_delta)

def quality(ratings):
    return global_env().quality(ratings)

def rate_1vs1(rating1, rating2, drawn=False, min_delta=DELTA):
    # We can treat it as free-for-all with 2 players
    if drawn:
        ranks = [0, 0]
    else:
        ranks = [0, 1]  # rating1 is winner, rating2 is loser
    res = global_env().rate([rating1, rating2], ranks, min_delta)
    return res[0], res[1]

def quality_1vs1(rating1, rating2):
    return global_env().quality([rating1, rating2])

def expose(rating):
    return global_env().expose(rating)