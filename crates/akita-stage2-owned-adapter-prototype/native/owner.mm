#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "../include/stage2.h"
#include "shader_source.h"
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <new>
#include <pthread.h>
#include <vector>
#include <cstddef>
using Ext=AkitaStage2Ext;
constexpr uint64_t Prime=UINT64_C(0xffffffffffffffc5),MaxN=UINT64_C(1)<<26,MaxCap=UINT64_C(8)<<30;
struct Params {uint64_t lanes,coefficients,first_len,second_len,skip_linear;Ext rho;uint64_t groups;};
struct CompactParams {Params p;uint64_t basis;Ext r1;};
struct AddParams {uint64_t live,domain,count,groups;Ext beta;};
static_assert(sizeof(Ext)==16 && alignof(Ext)==8 && offsetof(Ext,c1)==8);
static_assert(sizeof(AkitaStage2Source)==16 && sizeof(AkitaStage2Reference)==32 && offsetof(AkitaStage2Reference,factor)==16);
static_assert(sizeof(AkitaStage2AdditionalPair)==72 && offsetof(AkitaStage2AdditionalPair,binary1)==56);
static_assert(sizeof(AkitaStage2Config)==40 && sizeof(AkitaStage2Round)==224 && alignof(AkitaStage2Round)==8 && offsetof(AkitaStage2Round,alpha)==48 && offsetof(AkitaStage2Round,binary_batching)==192 && offsetof(AkitaStage2Round,additional_pairs)==208);
static_assert(sizeof(Params)==64 && sizeof(CompactParams)==88 && sizeof(AddParams)==48 && sizeof(AkitaStage2Message)==160);
#define ABI_OFFSET(T,F,N) static_assert(offsetof(T,F)==N)
static_assert(sizeof(void*)==8 && alignof(AkitaStage2Config)==8 && alignof(AkitaStage2Source)==8 && alignof(AkitaStage2Reference)==8 && alignof(AkitaStage2AdditionalPair)==8 && alignof(AkitaStage2Message)==8);
ABI_OFFSET(AkitaStage2Config,lanes,0);ABI_OFFSET(AkitaStage2Config,coefficients,8);ABI_OFFSET(AkitaStage2Config,basis,16);ABI_OFFSET(AkitaStage2Config,max_payload_bytes,24);ABI_OFFSET(AkitaStage2Config,initial_domain_len,32);
ABI_OFFSET(AkitaStage2Ext,c0,0);ABI_OFFSET(AkitaStage2Ext,c1,8);
ABI_OFFSET(AkitaStage2Source,offset,0);ABI_OFFSET(AkitaStage2Source,lanes,8);
ABI_OFFSET(AkitaStage2Reference,source,0);ABI_OFFSET(AkitaStage2Reference,lane,8);ABI_OFFSET(AkitaStage2Reference,factor,16);
ABI_OFFSET(AkitaStage2AdditionalPair,parent,0);ABI_OFFSET(AkitaStage2AdditionalPair,linear0,8);ABI_OFFSET(AkitaStage2AdditionalPair,linear1,24);ABI_OFFSET(AkitaStage2AdditionalPair,binary0,40);ABI_OFFSET(AkitaStage2AdditionalPair,binary1,56);
ABI_OFFSET(AkitaStage2Round,input_coefficients,0);ABI_OFFSET(AkitaStage2Round,skip_linear,8);ABI_OFFSET(AkitaStage2Round,r0,16);ABI_OFFSET(AkitaStage2Round,r1,32);
ABI_OFFSET(AkitaStage2Round,alpha,48);ABI_OFFSET(AkitaStage2Round,alpha_len,56);ABI_OFFSET(AkitaStage2Round,lane_weights,64);ABI_OFFSET(AkitaStage2Round,lane_weights_len,72);ABI_OFFSET(AkitaStage2Round,eq_first,80);ABI_OFFSET(AkitaStage2Round,eq_first_len,88);ABI_OFFSET(AkitaStage2Round,eq_second,96);ABI_OFFSET(AkitaStage2Round,eq_second_len,104);ABI_OFFSET(AkitaStage2Round,sources,112);ABI_OFFSET(AkitaStage2Round,sources_len,120);ABI_OFFSET(AkitaStage2Round,source_records,128);ABI_OFFSET(AkitaStage2Round,source_records_len,136);ABI_OFFSET(AkitaStage2Round,lane_offsets,144);ABI_OFFSET(AkitaStage2Round,lane_offsets_len,152);ABI_OFFSET(AkitaStage2Round,references,160);ABI_OFFSET(AkitaStage2Round,references_len,168);ABI_OFFSET(AkitaStage2Round,additional_domain_len,176);ABI_OFFSET(AkitaStage2Round,additional_live_len,184);ABI_OFFSET(AkitaStage2Round,binary_batching,192);ABI_OFFSET(AkitaStage2Round,additional_pairs,208);ABI_OFFSET(AkitaStage2Round,additional_pairs_len,216);
ABI_OFFSET(AkitaStage2Message,ordinary,0);ABI_OFFSET(AkitaStage2Message,additional,96);
#undef ABI_OFFSET
struct Error {uint32_t status;};
static void need(bool b,uint32_t status=AKITA_STAGE2_DEVICE){if(!b)throw Error{status};}
static bool power2(uint64_t n){return n && !(n&(n-1));}
static bool canonical(Ext x){return x.c0<Prime && x.c1<Prime;}
static bool extspan(const Ext* p,uint64_t n){if(n>MaxN || (n && !p))return false;for(uint64_t i=0;i<n;i++)if(!canonical(p[i]))return false;return true;}
static bool add_bytes(uint64_t& total,uint64_t count,uint64_t width,uint64_t cap){if(count>cap/width)return false;uint64_t n=count*width;if(n>cap-total)return false;total+=n;return true;}
static uint64_t initial_bytes(uint64_t n){return n+16*(n/4+n/8)+4096*16+6*1024*16+96+8*512;}
static uint32_t admission(const AkitaStage2Config* c,const int8_t* p,uint64_t n){
    if(!c || !c->lanes || !power2(c->coefficients) || c->coefficients<8 || c->lanes>MaxN/c->coefficients || (c->basis!=4 && c->basis!=8 && c->basis!=16 && c->basis!=32 && c->basis!=64) || c->max_payload_bytes>MaxCap || n!=c->lanes*c->coefficients || !p)return AKITA_STAGE2_INVALID;
    if(initial_bytes(n)>c->max_payload_bytes || !power2(c->initial_domain_len) || c->initial_domain_len<n || c->initial_domain_len>MaxN)return AKITA_STAGE2_INVALID;
    for(uint64_t i=0;i<n;i++)if(p[i]<-int(c->basis/2) || p[i]>=int(c->basis/2))return AKITA_STAGE2_INVALID;
    return AKITA_STAGE2_OK;
}
struct Buffer {
    id<MTLBuffer> value;uint64_t bytes;
    Buffer(id<MTLDevice> dev,uint64_t n):bytes(n){need(n<=UINT64_MAX-512 && n+512<=dev.maxBufferLength,AKITA_STAGE2_INVALID);value=[dev newBufferWithLength:n+512 options:MTLResourceStorageModeShared];need(value!=nil && value.contents,AKITA_STAGE2_ALLOCATION);memset(value.contents,0xA5,n+512);memset(data(),0,n);}
    void* data()const{return (char*)value.contents+256;}
    void copy(const void* p){if(bytes)memcpy(data(),p,bytes);}
    void guard()const{const auto* p=(const unsigned char*)value.contents;for(uint64_t i=0;i<256;i++)need(p[i]==0xA5 && p[256+bytes+i]==0xA5);}
};
struct AkitaStage2Owner {
    enum State {Compact,Coefficients,Poisoned,Finished};State state=Compact;
    pthread_t thread=pthread_self();AkitaStage2Config config;uint64_t coefficients,domain,payload;unsigned current=1,spare=0;
    id<MTLDevice> device;id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> lutPipeline,entryPipeline,foldPipeline,reducePipeline,addPipeline,addReducePipeline;
    std::unique_ptr<Buffer> witness[2],compact,lut,partials,messages;
    AkitaStage2Owner(const AkitaStage2Config& c,const int8_t* input):config(c),coefficients(c.coefficients),domain(c.initial_domain_len),payload(initial_bytes(c.lanes*c.coefficients)){
        device=MTLCreateSystemDefaultDevice();need(device!=nil);NSError* error=nil;
        id<MTLLibrary> lib=[device newLibraryWithSource:[NSString stringWithUTF8String:shader_source] options:nil error:&error];need(lib!=nil);
        auto pipeline=[&](NSString* name){id<MTLComputePipelineState> p=[device newComputePipelineStateWithFunction:[lib newFunctionWithName:name] error:&error];need(p!=nil && p.maxTotalThreadsPerThreadgroup>=256 && p.staticThreadgroupMemoryLength<=device.maxThreadgroupMemoryLength);return p;};
        lutPipeline=pipeline(@"compact_lut");entryPipeline=pipeline(@"stage2_compact");foldPipeline=pipeline(@"stage2_fold");reducePipeline=pipeline(@"stage2_reduce");addPipeline=pipeline(@"stage2_additional");addReducePipeline=pipeline(@"stage2_additional_reduce");queue=[device newCommandQueue];need(queue!=nil);
        uint64_t n=c.lanes*c.coefficients;compact=std::make_unique<Buffer>(device,n);compact->copy(input);
        witness[0]=std::make_unique<Buffer>(device,n/4*16);witness[1]=std::make_unique<Buffer>(device,n/8*16);
        lut=std::make_unique<Buffer>(device,4096*16);partials=std::make_unique<Buffer>(device,6*1024*16);messages=std::make_unique<Buffer>(device,96);
    }
    bool same_thread()const{return pthread_equal(thread,pthread_self());}
    uint32_t validate(const AkitaStage2Round* r,bool entry)const{
        if(!same_thread())return AKITA_STAGE2_THREAD;
        if(state!=(entry?Compact:Coefficients))return AKITA_STAGE2_STATE;
        if(!r || r->input_coefficients!=coefficients || coefficients<(entry?8:4) || r->skip_linear>1 || !canonical(r->r0) || !canonical(r->r1) || (!entry && (r->r1.c0||r->r1.c1)))return AKITA_STAGE2_INVALID;
        uint64_t m=coefficients/(entry?4:2),live=config.lanes*m,pairs=live/2;
        if(r->alpha_len!=m || r->lane_weights_len!=config.lanes || !power2(r->eq_first_len) || r->eq_first_len>pairs || r->eq_second_len!=(pairs+r->eq_first_len-1)/r->eq_first_len || r->lane_offsets_len!=config.lanes+1)return AKITA_STAGE2_INVALID;
        if(!extspan(r->alpha,r->alpha_len)||!extspan(r->lane_weights,r->lane_weights_len)||!extspan(r->eq_first,r->eq_first_len)||!extspan(r->eq_second,r->eq_second_len)||!extspan(r->sources,r->sources_len))return AKITA_STAGE2_INVALID;
        if(r->source_records_len>1048576 || r->references_len>1048576 || r->additional_pairs_len>1048576 || (r->source_records_len&&!r->source_records)||(r->references_len&&!r->references)||!r->lane_offsets||(r->additional_pairs_len&&!r->additional_pairs))return AKITA_STAGE2_INVALID;
        uint64_t position=0;for(uint64_t i=0;i<r->source_records_len;i++){auto x=r->source_records[i];if(x.offset!=position || x.lanes>r->sources_len/m || x.lanes*m>r->sources_len-position)return AKITA_STAGE2_INVALID;position+=x.lanes*m;}if(position!=r->sources_len)return AKITA_STAGE2_INVALID;
        if(r->lane_offsets[0]!=0 || r->lane_offsets[config.lanes]!=r->references_len)return AKITA_STAGE2_INVALID;
        for(uint64_t i=1;i<=config.lanes;i++)if(r->lane_offsets[i]<r->lane_offsets[i-1]||r->lane_offsets[i]>r->references_len)return AKITA_STAGE2_INVALID;
        for(uint64_t i=0;i<r->references_len;i++){auto x=r->references[i];if(x.source>=r->source_records_len||x.lane>=r->source_records[x.source].lanes||!canonical(x.factor))return AKITA_STAGE2_INVALID;}
        if(r->additional_live_len!=live || r->additional_domain_len!=domain/(entry?4:2) || r->additional_domain_len>MaxN || !canonical(r->binary_batching))return AKITA_STAGE2_INVALID;
        uint64_t prev=0;for(uint64_t i=0;i<r->additional_pairs_len;i++){auto x=r->additional_pairs[i];if(x.parent>=r->additional_domain_len/2||(i&&x.parent<=prev)||!canonical(x.linear0)||!canonical(x.linear1)||!canonical(x.binary0)||!canonical(x.binary1))return AKITA_STAGE2_INVALID;prev=x.parent;}
        uint64_t total=payload,cap=config.max_payload_bytes;
        for(auto n:{r->alpha_len,r->lane_weights_len,r->eq_first_len,r->eq_second_len,r->sources_len})if(!add_bytes(total,n,16,cap))return AKITA_STAGE2_INVALID;
        if(!add_bytes(total,r->source_records_len,16,cap)||!add_bytes(total,r->lane_offsets_len,8,cap)||!add_bytes(total,r->references_len,32,cap)||!add_bytes(total,r->additional_pairs_len,72,cap)||!add_bytes(total,4*1024+4,16,cap)||!add_bytes(total,11,512,cap))return AKITA_STAGE2_INVALID;
        return AKITA_STAGE2_OK;
    }
    void execute(const AkitaStage2Round& r,bool entry,AkitaStage2Message* output){
        state=Poisoned;uint64_t m=coefficients/(entry?4:2),live=config.lanes*m,groups=std::min<uint64_t>((live/2+255)/256,1024);
        Params p{config.lanes,coefficients,r.eq_first_len,r.eq_second_len,r.skip_linear,r.r0,groups};CompactParams cp{p,config.basis,r.r1};
        AddParams ap{live,r.additional_domain_len,r.additional_pairs_len,std::max<uint64_t>(1,std::min<uint64_t>((r.additional_pairs_len+255)/256,1024)),r.binary_batching};
        std::array<uint64_t,11> sizes={r.alpha_len*16,r.lane_weights_len*16,r.eq_first_len*16,r.eq_second_len*16,r.sources_len*16,r.source_records_len*16,r.lane_offsets_len*8,r.references_len*32,r.additional_pairs_len*72,ap.groups*64,64};
        std::array<const void*,9> inputs={r.alpha,r.lane_weights,r.eq_first,r.eq_second,r.sources,r.source_records,r.lane_offsets,r.references,r.additional_pairs};
        std::vector<Buffer> temp;temp.reserve(11);for(size_t i=0;i<11;i++){temp.emplace_back(device,sizes[i]);if(i<9)temp.back().copy(inputs[i]);}
        id<MTLCommandBuffer> cmd=[queue commandBuffer];need(cmd!=nil);id<MTLComputeCommandEncoder> enc;
        if(entry){enc=[cmd computeCommandEncoder];need(enc!=nil);[enc setComputePipelineState:lutPipeline];[enc setBuffer:lut->value offset:256 atIndex:0];[enc setBytes:&cp length:sizeof(cp) atIndex:1];[enc dispatchThreadgroups:MTLSizeMake(config.basis==4?1:(config.basis==8?16:(config.basis*config.basis+255)/256),1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];}
        enc=[cmd computeCommandEncoder];need(enc!=nil);[enc setComputePipelineState:(entry?entryPipeline:foldPipeline)];
        [enc setBuffer:(entry?compact->value:witness[current]->value) offset:256 atIndex:0];
        for(size_t i=0;i<5;i++)[enc setBuffer:temp[i].value offset:256 atIndex:i+1];
        [enc setBuffer:witness[spare]->value offset:256 atIndex:6];[enc setBuffer:partials->value offset:256 atIndex:7];
        if(entry){[enc setBytes:&cp length:sizeof(cp) atIndex:8];[enc setBuffer:lut->value offset:256 atIndex:12];}else [enc setBytes:&p length:sizeof(p) atIndex:8];
        for(size_t i=5;i<8;i++)[enc setBuffer:temp[i].value offset:256 atIndex:i+4];
        [enc dispatchThreadgroups:MTLSizeMake(groups,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];
        enc=[cmd computeCommandEncoder];need(enc!=nil);[enc setComputePipelineState:reducePipeline];[enc setBuffer:partials->value offset:256 atIndex:0];[enc setBuffer:messages->value offset:256 atIndex:1];[enc setBytes:&groups length:8 atIndex:2];[enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];
        enc=[cmd computeCommandEncoder];need(enc!=nil);[enc setComputePipelineState:addPipeline];[enc setBuffer:witness[spare]->value offset:256 atIndex:0];[enc setBuffer:temp[8].value offset:256 atIndex:1];[enc setBuffer:temp[9].value offset:256 atIndex:2];[enc setBytes:&ap length:sizeof(ap) atIndex:3];[enc dispatchThreadgroups:MTLSizeMake(ap.groups,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];
        enc=[cmd computeCommandEncoder];need(enc!=nil);[enc setComputePipelineState:addReducePipeline];[enc setBuffer:temp[9].value offset:256 atIndex:0];[enc setBuffer:temp[10].value offset:256 atIndex:1];[enc setBytes:&ap.groups length:8 atIndex:2];[enc dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[enc endEncoding];
        [cmd commit];[cmd waitUntilCompleted];need(cmd.status==MTLCommandBufferStatusCompleted);
        for(auto& x:temp)x.guard();for(auto& x:witness)x->guard();compact->guard();lut->guard();partials->guard();messages->guard();
        AkitaStage2Message result;memcpy(result.ordinary,messages->data(),96);memcpy(result.additional,temp[10].data(),64);
        for(auto x:result.ordinary)need(canonical(x));for(auto x:result.additional)need(canonical(x));
        std::swap(current,spare);coefficients=m;domain/=(entry?4:2);state=Coefficients;memcpy(output,&result,sizeof(result));
    }
};
extern "C" uint32_t akita_stage2_admit(const AkitaStage2Config* c,const int8_t* p,uint64_t n){return admission(c,p,n);}
extern "C" uint32_t akita_stage2_create(const AkitaStage2Config* c,const int8_t* p,uint64_t n,AkitaStage2Owner** out){
    if(!out)return AKITA_STAGE2_INVALID;*out=nullptr;uint32_t status=admission(c,p,n);if(status)return status;
    @autoreleasepool { @try {try{auto value=std::make_unique<AkitaStage2Owner>(*c,p);*out=value.release();return AKITA_STAGE2_OK;}
    catch(const Error& e){return e.status;}catch(const std::bad_alloc&){return AKITA_STAGE2_ALLOCATION;}catch(...){return AKITA_STAGE2_INTERNAL;}}
    @catch(NSException* exception){(void)exception;return AKITA_STAGE2_DEVICE;} }
}
static uint32_t operate(AkitaStage2Owner* o,const AkitaStage2Round* r,AkitaStage2Message* out,bool entry){
    if(!o||!out)return AKITA_STAGE2_INVALID;uint32_t status=o->validate(r,entry);if(status)return status;
    @autoreleasepool { @try {try{o->execute(*r,entry,out);return AKITA_STAGE2_OK;}
    catch(const Error& e){o->state=AkitaStage2Owner::Poisoned;return e.status==AKITA_STAGE2_INVALID?AKITA_STAGE2_DEVICE:e.status;}
    catch(const std::bad_alloc&){o->state=AkitaStage2Owner::Poisoned;return AKITA_STAGE2_ALLOCATION;}catch(...){o->state=AkitaStage2Owner::Poisoned;return AKITA_STAGE2_INTERNAL;}}
    @catch(NSException* exception){(void)exception;o->state=AkitaStage2Owner::Poisoned;return AKITA_STAGE2_DEVICE;} }
}
extern "C" uint32_t akita_stage2_compact_entry(AkitaStage2Owner* o,const AkitaStage2Round* r,AkitaStage2Message* out){return operate(o,r,out,true);}
extern "C" uint32_t akita_stage2_advance(AkitaStage2Owner* o,const AkitaStage2Round* r,AkitaStage2Message* out){return operate(o,r,out,false);}
extern "C" uint32_t akita_stage2_finish(AkitaStage2Owner* o,Ext* out,uint64_t count){
    if(!o)return AKITA_STAGE2_INVALID;if(!o->same_thread())return AKITA_STAGE2_THREAD;if(o->state!=AkitaStage2Owner::Coefficients)return AKITA_STAGE2_STATE;
    if(!out || count!=o->config.lanes*o->coefficients)return AKITA_STAGE2_INVALID;
    @autoreleasepool { @try {try{o->state=AkitaStage2Owner::Poisoned;o->witness[o->current]->guard();auto* fields=(const Ext*)o->witness[o->current]->data();for(uint64_t i=0;i<count;i++)need(canonical(fields[i]));memcpy(out,fields,count*16);o->state=AkitaStage2Owner::Finished;return AKITA_STAGE2_OK;}
    catch(const Error& e){return e.status;}catch(const std::bad_alloc&){return AKITA_STAGE2_ALLOCATION;}catch(...){return AKITA_STAGE2_INTERNAL;}}
    @catch(NSException* exception){(void)exception;return AKITA_STAGE2_DEVICE;} }
}
extern "C" uint32_t akita_stage2_destroy(AkitaStage2Owner* o){
    if(!o)return AKITA_STAGE2_OK;if(!o->same_thread())return AKITA_STAGE2_THREAD;
    @autoreleasepool { @try {try{delete o;return AKITA_STAGE2_OK;}catch(...){return AKITA_STAGE2_INTERNAL;}}@catch(NSException* exception){(void)exception;return AKITA_STAGE2_DEVICE;} }
}
