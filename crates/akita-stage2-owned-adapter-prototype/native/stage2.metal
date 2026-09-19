// field.h is prepended byte-for-byte. All inputs are canonical Ext limbs.
struct Stage2Params { ulong lanes, coefficients, first_len, second_len, skip_linear; Ext rho; ulong groups; };
struct SourceRecord { ulong offset,lanes; };
struct Reference { ulong source,lane; Ext factor; };
struct Six { Ext v[6]; };
inline Six zero_six() { Six x; for(uint k=0;k<6;k++) x.v[k]=Ext{0,0}; return x; }
inline void sum_six(Six value,threadgroup Ext* scratch,device Ext* out,uint tid,ulong group) {
    for(uint k=0;k<6;k++) scratch[k*256+tid]=value.v[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=128;stride;stride>>=1) {
        if(tid<stride) for(uint k=0;k<6;k++) scratch[k*256+tid]=ext_add(scratch[k*256+tid],scratch[k*256+tid+stride]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0) for(uint k=0;k<6;k++) out[group*6+k]=scratch[k*256];
}
kernel void stage2_fold(
    device const Ext* source [[buffer(0)]],device const Ext* alpha [[buffer(1)]],
    device const Ext* lane_weights [[buffer(2)]],device const Ext* first [[buffer(3)]],
    device const Ext* second [[buffer(4)]],device const Ext* sources [[buffer(5)]],
    device Ext* output [[buffer(6)]],device Ext* partials [[buffer(7)]],
    constant Stage2Params& p [[buffer(8)]],device const SourceRecord* records [[buffer(9)]],
    device const ulong* offsets [[buffer(10)]],device const Reference* refs [[buffer(11)]],uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup Ext scratch[1536];
    Six sums=zero_six();
    const ulong m=p.coefficients/2, pairs_per_lane=m/2, count=p.lanes*pairs_per_lane;
    for(ulong j=ulong(group)*256+tid;j<count;j+=p.groups*256) {
        const ulong lane=j/pairs_per_lane,k=j%pairs_per_lane;
        const ulong s=lane*p.coefficients+4*k,o=lane*m+2*k;
        const Ext w0=ext_add(source[s],ext_mul(p.rho,ext_sub(source[s+1],source[s])));
        const Ext w1=ext_add(source[s+2],ext_mul(p.rho,ext_sub(source[s+3],source[s+2])));
        output[o]=w0;output[o+1]=w1;
        const Ext dw=ext_sub(w1,w0),one=Ext{1,0};
        const Ext e=ext_mul(first[j&(p.first_len-1)],second[j/p.first_len]);
        sums.v[0]=ext_add(sums.v[0],ext_mul(e,ext_mul(w0,ext_add(w0,one))));
        if(!p.skip_linear) sums.v[1]=ext_add(sums.v[1],ext_mul(e,ext_mul(dw,ext_add(ext_add(w0,w0),one))));
        sums.v[2]=ext_add(sums.v[2],ext_mul(e,ext_mul(dw,dw)));
        Ext t0=Ext{0,0},t1=Ext{0,0};
        for(ulong r=offsets[lane];r<offsets[lane+1];r++) {
            const Reference ref=refs[r];const SourceRecord rec=records[ref.source];
            const ulong base=rec.offset+ref.lane*m+2*k;
            t0=ext_add(t0,ext_mul(ref.factor,sources[base]));
            t1=ext_add(t1,ext_mul(ref.factor,sources[base+1]));
        }
        const Ext p0=ext_add(ext_mul(alpha[2*k],lane_weights[lane]),t0);
        const Ext p1=ext_add(ext_mul(alpha[2*k+1],lane_weights[lane]),t1);
        const Ext dp=ext_sub(p1,p0);
        sums.v[3]=ext_add(sums.v[3],ext_mul(w0,p0));
        sums.v[4]=ext_add(sums.v[4],ext_add(ext_mul(w0,dp),ext_mul(dw,p0)));
        sums.v[5]=ext_add(sums.v[5],ext_mul(dw,dp));
    }
    sum_six(sums,scratch,partials,tid,group);
}
kernel void stage2_reduce(device const Ext* partials [[buffer(0)]],device Ext* output [[buffer(1)]],
    constant ulong& groups [[buffer(2)]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup Ext scratch[1536];Six sums=zero_six();
    for(ulong j=tid;j<groups;j+=256) for(uint k=0;k<6;k++) sums.v[k]=ext_add(sums.v[k],partials[j*6+k]);
    sum_six(sums,scratch,output,tid,0);
}

struct AddPair { ulong parent; Ext l0,l1,b0,b1; };
struct AddParams { ulong live,domain,count,groups;Ext beta; };
inline void add_sum(thread Ext* v,threadgroup Ext* scratch,device Ext* out,uint tid,ulong group) {
    for(uint k=0;k<4;k++)scratch[k*256+tid]=v[k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint step=128;step;step>>=1){if(tid<step)for(uint k=0;k<4;k++)scratch[k*256+tid]=ext_add(scratch[k*256+tid],scratch[k*256+tid+step]);threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(tid==0)for(uint k=0;k<4;k++)out[group*4+k]=scratch[k*256];
}
kernel void stage2_additional(device const Ext* witness [[buffer(0)]],device const AddPair* pairs [[buffer(1)]],device Ext* partials [[buffer(2)]],constant AddParams& p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]) {
    threadgroup Ext scratch[1024];Ext v[4]={Ext{0,0},Ext{0,0},Ext{0,0},Ext{0,0}};
    for(ulong i=ulong(group)*256+tid;i<p.count;i+=p.groups*256){
        AddPair a=pairs[i];ulong j=2*a.parent;
        Ext w=j<p.live?witness[j]:Ext{0,0},w1=j+1<p.live?witness[j+1]:Ext{0,0};
        Ext dw=ext_sub(w1,w),dl=ext_sub(a.l1,a.l0),db=ext_sub(a.b1,a.b0);
        Ext q0=ext_mul(w,ext_add(w,Ext{1,0})),q1=ext_mul(dw,ext_add(ext_add(w,w),Ext{1,0})),q2=ext_mul(dw,dw);
        v[0]=ext_add(v[0],ext_add(ext_mul(w,a.l0),ext_mul(p.beta,ext_mul(a.b0,q0))));
        v[1]=ext_add(v[1],ext_add(ext_add(ext_mul(w,dl),ext_mul(dw,a.l0)),ext_mul(p.beta,ext_add(ext_mul(a.b0,q1),ext_mul(db,q0)))));
        v[2]=ext_add(v[2],ext_add(ext_mul(dw,dl),ext_mul(p.beta,ext_add(ext_mul(a.b0,q2),ext_mul(db,q1)))));
        v[3]=ext_add(v[3],ext_mul(p.beta,ext_mul(db,q2)));
    }
    add_sum(v,scratch,partials,tid,group);
}
kernel void stage2_additional_reduce(device const Ext* partials [[buffer(0)]],device Ext* output [[buffer(1)]],constant ulong& groups [[buffer(2)]],uint tid [[thread_index_in_threadgroup]]) {
    threadgroup Ext scratch[1024];Ext v[4]={Ext{0,0},Ext{0,0},Ext{0,0},Ext{0,0}};
    for(ulong i=tid;i<groups;i+=256)for(uint k=0;k<4;k++)v[k]=ext_add(v[k],partials[i*4+k]);
    add_sum(v,scratch,output,tid,0);
}

struct CompactParams { ulong lanes, coefficients, first_len, second_len, skip_linear; Ext rho; ulong groups; ulong basis; Ext r1; };
kernel void stage2_compact(
    device const char* source [[buffer(0)]],device const Ext* alpha [[buffer(1)]],
    device const Ext* lane_weights [[buffer(2)]],device const Ext* first [[buffer(3)]],
    device const Ext* second [[buffer(4)]],device const Ext* sources [[buffer(5)]],
    device Ext* output [[buffer(6)]],device Ext* partials [[buffer(7)]],
    constant CompactParams& p [[buffer(8)]],device const SourceRecord* records [[buffer(9)]],
    device const ulong* offsets [[buffer(10)]],device const Reference* refs [[buffer(11)]],device const Ext* lut [[buffer(12)]],uint tid [[thread_index_in_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {
    threadgroup Ext scratch[1536];
    Six sums=zero_six();
    const ulong m=p.coefficients/4, pairs_per_lane=m/2, count=p.lanes*pairs_per_lane;
    for(ulong j=ulong(group)*256+tid;j<count;j+=p.groups*256) {
        const ulong lane=j/pairs_per_lane,k=j%pairs_per_lane;
        const ulong s=lane*p.coefficients+8*k,o=lane*m+2*k;
        Ext w0,w1;
        if(p.basis<=8) {
        const uint bits=p.basis==4?2:3;
        uint i0=0,i1=0;
        for(uint q=0;q<4;q++) {
            i0|=uint(int(source[s+q])+int(p.basis/2))<<(bits*q);
            i1|=uint(int(source[s+4+q])+int(p.basis/2))<<(bits*q);
        }
        w0=lut[i0];w1=lut[i1];
        } else {
            // Wide balanced digits use b² first-challenge pairs, not b⁴ quads.
            const uint bits=p.basis==16?4:(p.basis==32?5:6), digit_bias=uint(p.basis/2);
            uint index[4];
            for(uint pair=0;pair<4;pair++) {
                const uint x=uint(int(source[s+2*pair])+int(digit_bias));
                const uint y=uint(int(source[s+2*pair+1])+int(digit_bias));
                index[pair]=x|(y<<bits);
            }
            const Ext a0=lut[index[0]],a1=lut[index[1]];
            const Ext b0=lut[index[2]],b1=lut[index[3]];
            w0=ext_add(a0,ext_mul(p.r1,ext_sub(a1,a0)));
            w1=ext_add(b0,ext_mul(p.r1,ext_sub(b1,b0)));
        }
        output[o]=w0;output[o+1]=w1;
        const Ext dw=ext_sub(w1,w0),one=Ext{1,0};
        const Ext e=ext_mul(first[j&(p.first_len-1)],second[j/p.first_len]);
        sums.v[0]=ext_add(sums.v[0],ext_mul(e,ext_mul(w0,ext_add(w0,one))));
        if(!p.skip_linear) sums.v[1]=ext_add(sums.v[1],ext_mul(e,ext_mul(dw,ext_add(ext_add(w0,w0),one))));
        sums.v[2]=ext_add(sums.v[2],ext_mul(e,ext_mul(dw,dw)));
        Ext t0=Ext{0,0},t1=Ext{0,0};
        for(ulong r=offsets[lane];r<offsets[lane+1];r++) {
            const Reference ref=refs[r];const SourceRecord rec=records[ref.source];
            const ulong base=rec.offset+ref.lane*m+2*k;
            t0=ext_add(t0,ext_mul(ref.factor,sources[base]));
            t1=ext_add(t1,ext_mul(ref.factor,sources[base+1]));
        }
        const Ext p0=ext_add(ext_mul(alpha[2*k],lane_weights[lane]),t0);
        const Ext p1=ext_add(ext_mul(alpha[2*k+1],lane_weights[lane]),t1);
        const Ext dp=ext_sub(p1,p0);
        sums.v[3]=ext_add(sums.v[3],ext_mul(w0,p0));
        sums.v[4]=ext_add(sums.v[4],ext_add(ext_mul(w0,dp),ext_mul(dw,p0)));
        sums.v[5]=ext_add(sums.v[5],ext_mul(dw,dp));
    }
    sum_six(sums,scratch,partials,tid,group);
}

inline Ext signed_digit(int d) {return Ext{d<0?0xffffffffffffffc5ul-ulong(-d):ulong(d),0};}
kernel void compact_lut(device Ext* lut [[buffer(0)]],constant CompactParams& p [[buffer(1)]],uint idx [[thread_position_in_grid]]) {
    if(p.basis>8) {
        const uint bits=p.basis==16?4:(p.basis==32?5:6),count=uint(p.basis*p.basis);
        if(idx>=count)return;
        const Ext a=signed_digit(int(idx&uint(p.basis-1))-int(p.basis/2));
        const Ext b=signed_digit(int(idx>>bits)-int(p.basis/2));
        lut[idx]=ext_add(a,ext_mul(p.rho,ext_sub(b,a)));
        return;
    }
    const uint bits=p.basis==4?2:3,count=1u<<(4*bits);if(idx>=count)return;
    Ext q[4];for(uint k=0;k<4;k++)q[k]=signed_digit(int((idx>>(bits*k))&uint(p.basis-1))-int(p.basis/2));
    const Ext a=ext_add(q[0],ext_mul(p.rho,ext_sub(q[1],q[0])));
    const Ext b=ext_add(q[2],ext_mul(p.rho,ext_sub(q[3],q[2])));
    lut[idx]=ext_add(a,ext_mul(p.r1,ext_sub(b,a)));
}
