//
//  LinearAlgebra.swift
//  MathSwift
//
//  Created by Enyan HUANG on 6/11/14.
//  Copyright (c) 2014 The Hong Kong Polytechnic University. All rights reserved.
//
//  Contributed by Dang Vinh on 09/10/2016
//  Added functions:
//  - M.qrDecomposition(): factorize matrix M = Q*R by using QR decomposition algorithm
//  - A.solve(B): find x of linear system Ax = B. If A is singular, return least square solution

import Foundation
import Accelerate

extension Matrix {
    
    public func eigen() -> (eigenvalues: [Double], eigenvectors: [Matrix]) {
        assert(self.rows == self.columns, "Square matrix is required to compute eigen values")
        var jobvl: Int8 = 78
        var jobvr: Int8 = 86
        var n = __CLPK_integer(self.rows)
        var a = self.transpose.elements
        var lda = n
        var wr = [Double](count: self.rows, repeatedValue: 0.0)
        var wi = [Double](count: self.rows, repeatedValue: 0.0)
        var vl = [Double](count: self.rows * self.rows, repeatedValue: 0.0)
        var ldvl = n
        var vr = [Double](count: self.rows * self.rows, repeatedValue: 0.0)
        var ldvr = n
        var lwork = n * 4
        var work = [Double](count: Int(lwork), repeatedValue: 0.0)
        var info: __CLPK_integer = 0
        
        dgeev_(&jobvl, &jobvr, &n, &a, &lda, &wr, &wi, &vl, &ldvl, &vr, &ldvr, &work, &lwork, &info)
        
        var eigenvectors = [Matrix](count: self.rows, repeatedValue: Matrix(rows: self.rows, columns: 1))
        for i in 0..<self.rows {
            for j in 0..<self.rows {
                eigenvectors[i].elements[j] = vr[i * self.rows + j]
            }
            
        }
        
        return (wr, eigenvectors)
    }
    
    public func singularValueDecomposition() -> (U: Matrix, S: Matrix, VT: Matrix) {
        
        var jobu: Int8 = 65
        var jobvt: Int8 = 65
        var m = __CLPK_integer(self.rows)
        var n = __CLPK_integer(self.columns)
        let minMN = min(m, n)
        var a = self.transpose.elements
        var lda = m
        var ldu = m
        var ldvt = n
        var lwork = __CLPK_integer(5 * self.rows * self.columns)
        var info: __CLPK_integer = 0
        var work = [Double](count: Int(lwork), repeatedValue: 0.0)
        var u = [Double](count: self.rows * self.rows, repeatedValue: 0.0)
        var s = [Double](count: Int(minMN), repeatedValue: 0.0)
        var vt = [Double](count: self.columns * self.columns, repeatedValue: 0.0)
        
        dgesvd_(&jobu, &jobvt, &m, &n, &a, &lda, &s, &u, &ldu, &vt, &ldvt, &work, &lwork, &info)
        
        let U = Matrix(rows: self.rows, columns: self.rows, elements: u)
        var S = Matrix(rows: self.rows, columns: self.columns)
        for i in 0..<Int(minMN) {
            S[i,i] = s[i].toMatrix()
        }
        let VT = Matrix(rows: self.columns, columns: self.columns, elements: vt)
        
        return (U.transpose, S, VT.transpose)
    }
    
    public func qrDecomposition() -> (Q: Matrix, R: Matrix) {
        var m = __CLPK_integer(self.rows)
        var n = __CLPK_integer(self.columns)
        let minMN = min(m, n)
        var a = self.transpose.elements
        var lda = m
        var tau = [Double](count: Int(minMN), repeatedValue: 0.0)
        var lwork = max(1,n)
        var work = [Double](count: Int(lwork), repeatedValue: 0.0)
        var info: __CLPK_integer = 0
        
        dgeqrf_(&m, &n, &a, &lda, &tau, &work, &lwork, &info)
        
        var R = Matrix(rows: Int(m), columns: Int(n))
        var Q = Matrix.identityWithSize(Int(m))
        
        let I = Matrix.identityWithSize(Int(m))
        for c in 0..<Int(n){
            var v = Matrix(rows: Int(m), columns: 1)
            for r in 0..<Int(m){
                if r <= c {
                    if r < Int(minMN){
                        R[r,c] = (-a[Int(m)*c+r]).toMatrix()
                    }
                    v[r,0] = 0.0
                }
                else {
                    v[r,0] = a[Int(m)*c+r].toMatrix()
                }
            }
            
            if c < Int(minMN){
                v[c,0] = 1.0
                let H = I-tau[c]*v*v.transpose
                Q = Q*H
            }
        }
        return (Q, R)
    }
}