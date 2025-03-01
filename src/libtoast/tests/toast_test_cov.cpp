
// Copyright (c) 2015-2019 by the parties listed in the AUTHORS file.
// All rights reserved.  Use of this source code is governed by
// a BSD-style license that can be found in the LICENSE file.

#include <toast_test.hpp>

#include <cmath>
#include <vector>


const int64_t TOASTcovTest::nsm = 2;
const int64_t TOASTcovTest::npix = 3;
const int64_t TOASTcovTest::nnz = 4;
const int64_t TOASTcovTest::nsamp = 100;
const int64_t TOASTcovTest::scale = 2.0;


TEST_F(TOASTcovTest, accumulate) {
    int64_t block = (int64_t)(nnz * (nnz + 1) / 2);

    std::vector <double> fakedata(nsm * npix * nnz);
    std::vector <int64_t> fakehits(nsm * npix);
    std::vector <double> fakeinvn(nsm * npix * block);

    std::vector <double> checkdata(nsm * npix * nnz);
    std::vector <int64_t> checkhits(nsm * npix);
    std::vector <double> checkinvn(nsm * npix * block);

    std::vector <double> signal(nsamp);
    std::vector <double> weights(nsamp * nnz);
    std::vector <int64_t> sm(nsamp);
    std::vector <int64_t> pix(nsamp);

    toast::rng_dist_normal(nsamp, 0, 0, 0, 0, signal.data());

    for (int64_t i = 0; i < (nsm * npix); ++i) {
        checkhits[i] = 0;
        fakehits[i] = 0;
        int64_t off = 0;
        for (int64_t k = 0; k < nnz; ++k) {
            checkdata[i * nnz + k] = 0.0;
            fakedata[i * nnz + k] = 0.0;
            for (int64_t m = k; m < nnz; ++m) {
                fakeinvn[i * block + off] = 0.0;
                checkinvn[i * block + off] = 0.0;
                off++;
            }
        }
    }

    for (int64_t i = 0; i < nsamp; ++i) {
        sm[i] = i % nsm;
        pix[i] = i % npix;
        for (int64_t k = 0; k < nnz; ++k) {
            weights[i * nnz + k] = (double)(k + 1);
        }
    }

    toast::cov_accum_diag(nsm, npix, nnz, nsamp, sm.data(), pix.data(),
                          weights.data(), scale, signal.data(),
                          fakedata.data(), fakehits.data(), fakeinvn.data());

    for (int64_t i = 0; i < nsamp; ++i) {
        checkhits[sm[i] * npix + pix[i]] += 1;
        int64_t off = 0;
        for (int64_t k = 0; k < nnz; ++k) {
            checkdata[sm[i] * (npix * nnz) + pix[i] * nnz + k] +=
                scale * signal[i] * weights[i * nnz + k];
            for (int64_t m = k; m < nnz; ++m) {
                checkinvn[sm[i] * (npix * block) + pix[i] * block + off] +=
                    scale * weights[i * nnz + k] * weights[i * nnz + m];
                off++;
            }
        }
    }

    for (int64_t i = 0; i < (nsm * npix); ++i) {
        // printf("%ld: %ld %ld\n", i, checkhits[i], fakehits[i]);
        EXPECT_EQ(checkhits[i], fakehits[i]);
        int64_t off = 0;
        for (int64_t k = 0; k < nnz; ++k) {
            // printf("  %ld: %0.12e %0.12e\n", k, checkdata[i*nnz+k],
            // fakedata[i*nnz+k]);
            EXPECT_FLOAT_EQ(checkdata[i * nnz + k], fakedata[i * nnz + k]);
            for (int64_t m = k; m < nnz; ++m) {
                // printf("    %ld: %0.12e %0.12e\n", k,
                // checkinvn[i*block+off], fakeinvn[i*block+off]);
                EXPECT_FLOAT_EQ(checkinvn[i * block + off],
                                fakeinvn[i * block + off]);
                off++;
            }
        }
    }
}


TEST_F(TOASTcovTest, eigendecompose) {
    int64_t block = (int64_t)(nnz * (nnz + 1) / 2);

    std::vector <double> fakedata(nsm * npix * block);
    std::vector <double> checkdata(nsm * npix * block);

    std::vector <double> rowdata(nnz);
    std::vector <double> cond(nsm * npix);

    for (int64_t k = 0; k < nnz; ++k) {
        rowdata[k] = 10.0 * (nnz - k);
    }

    double threshold = 1.0e-6;

    for (int64_t i = 0; i < (nsm * npix); ++i) {
        int64_t off = 0;
        for (int64_t k = 0; k < nnz; ++k) {
            for (int64_t m = k; m < nnz; ++m) {
                fakedata[i * block + off] = rowdata[m - k];
                checkdata[i * block + off] = fakedata[i * block + off];
                off++;
            }
        }
    }

    toast::cov_eigendecompose_diag(nsm, npix, nnz, fakedata.data(), cond.data(),
                                   threshold, true);
    toast::cov_eigendecompose_diag(nsm, npix, nnz, fakedata.data(), cond.data(),
                                   threshold, true);

    for (int64_t i = 0; i < (nsm * npix); ++i) {
        int64_t off = 0;
        for (int64_t k = 0; k < nnz; ++k) {
            for (int64_t m = k; m < nnz; ++m) {
                EXPECT_FLOAT_EQ(checkdata[i * block + k],
                                fakedata[i * block + k]);
                off++;
            }
        }
    }
}


TEST_F(TOASTcovTest, matrixmultiply) {
    int64_t block = (int64_t)(nnz * (nnz + 1) / 2);

    std::vector <double> data1(nsm * npix * block);
    std::vector <double> data2(nsm * npix * block);

    std::vector <double> rowdata(nnz);
    std::vector <double> cond(nsm * npix);

    for (int64_t k = 0; k < nnz; ++k) {
        rowdata[k] = 10.0 * (nnz - k);
    }

    double threshold = 1.0e-6;

    for (int64_t i = 0; i < (nsm * npix); ++i) {
        int64_t off = 0;
        for (int64_t k = 0; k < nnz; ++k) {
            for (int64_t m = k; m < nnz; ++m) {
                data1[i * block + off] = rowdata[m - k];
                data2[i * block + off] = rowdata[m - k];
                off++;
            }
        }
    }

    toast::cov_eigendecompose_diag(nsm, npix, nnz,
                                   data2.data(), cond.data(), threshold, true);

    toast::cov_mult_diag(nsm, npix, nnz, data1.data(), data2.data());

    for (int64_t i = 0; i < (nsm * npix); ++i) {
        int64_t off = 0;
        for (int64_t k = 0; k < nnz; ++k) {
            for (int64_t m = k; m < nnz; ++m) {
                // printf("result [%ld, %ld] = %0.12e\n", i, off,
                // data1[i*block+off]);
                if (m == k) {
                    EXPECT_FLOAT_EQ(1.0, data1[i * block + off]);
                } else {
                    ASSERT_TRUE(::fabs(data1[i * block + off]) < 1.0e-12);
                }
                off++;
            }
        }
    }
}
