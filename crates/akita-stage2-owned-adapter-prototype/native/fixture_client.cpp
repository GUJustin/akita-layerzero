#include "../include/stage2.h"
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
static void check(bool b,const char* s){if(!b)throw std::runtime_error(s);}
static std::vector<uint8_t> file(const char* p){std::ifstream f(p,std::ios::binary|std::ios::ate);check(bool(f),"open");auto n=f.tellg();check(n>=0 && n<64*1024*1024,"file cap");std::vector<uint8_t> v((size_t)n);f.seekg(0);f.read((char*)v.data(),n);check(bool(f),"read");return v;}
struct Reader{const std::vector<uint8_t>& v;size_t at=0;uint64_t word(){check(v.size()-at>=8,"word bounds");uint64_t x=0;for(unsigned i=0;i<8;i++)x|=uint64_t(v[at++])<<(8*i);return x;}
std::vector<AkitaStage2Ext> ext(uint64_t n){check(n<=(v.size()-at)/16,"Ext bounds");std::vector<AkitaStage2Ext> a;for(uint64_t i=0;i<n;i++)a.push_back({word(),word()});return a;}};
struct Request{AkitaStage2Round r{};std::array<std::vector<AkitaStage2Ext>,5> f;std::vector<AkitaStage2Source> s;std::vector<uint64_t> o;std::vector<AkitaStage2Reference> refs;std::vector<AkitaStage2AdditionalPair> pairs;
void read(Reader& in,uint64_t L,uint64_t C,bool entry){
 r.input_coefficients=C;r.eq_first_len=in.word();r.eq_second_len=in.word();r.skip_linear=in.word();r.r0={in.word(),in.word()};r.r1=entry?AkitaStage2Ext{in.word(),in.word()}:AkitaStage2Ext{0,0};
 uint64_t ns=in.word(),nr=in.word(),nf=in.word();uint64_t m=C/(entry?4:2);r.alpha_len=m;r.lane_weights_len=L;r.sources_len=ns;r.source_records_len=nr;r.lane_offsets_len=L+1;r.references_len=nf;
 f[0]=in.ext(m);f[1]=in.ext(L);f[2]=in.ext(r.eq_first_len);f[3]=in.ext(r.eq_second_len);f[4]=in.ext(ns);
 check(nr<=(in.v.size()-in.at)/16,"records bounds");for(uint64_t i=0;i<nr;i++)s.push_back({in.word(),in.word()});check(L+1<=(in.v.size()-in.at)/8,"offset bounds");for(uint64_t i=0;i<=L;i++)o.push_back(in.word());
 check(nf<=(in.v.size()-in.at)/32,"reference bounds");for(uint64_t i=0;i<nf;i++)refs.push_back({in.word(),in.word(),{in.word(),in.word()}});
 r.additional_domain_len=in.word();r.additional_live_len=L*m;r.binary_batching={in.word(),in.word()};r.additional_pairs_len=in.word();check(r.additional_pairs_len<=(in.v.size()-in.at)/72,"pair bounds");
 for(uint64_t i=0;i<r.additional_pairs_len;i++)pairs.push_back({in.word(),{in.word(),in.word()},{in.word(),in.word()},{in.word(),in.word()},{in.word(),in.word()}});
 r.alpha=f[0].data();r.lane_weights=f[1].data();r.eq_first=f[2].data();r.eq_second=f[3].data();r.sources=f[4].data();r.source_records=s.data();r.lane_offsets=o.data();r.references=refs.data();r.additional_pairs=pairs.data();
}};
struct Lease{AkitaStage2Owner* p=nullptr;~Lease(){if(p)akita_stage2_destroy(p);}};
static void append(std::vector<uint8_t>& out,const AkitaStage2Ext* p,size_t n){for(size_t i=0;i<n;i++)for(uint64_t x:{p[i].c0,p[i].c1})for(unsigned j=0;j<8;j++)out.push_back(uint8_t(x>>(8*j)));}
int main(int argc,char** argv){try{
 check(argc==4,"usage: owned INPUT EXPECTED OUTPUT");auto raw=file(argv[1]),expected=file(argv[2]);Reader in{raw};check(in.word()==0x314f4753,"magic");uint64_t L=in.word(),C=in.word(),basis=in.word(),domain=in.word(),advances=in.word();
 check(L>0 && C>=8 && !(C&(C-1)) && L<=(UINT64_C(1)<<26)/C && advances<=24,"geometry");check((C/4>>advances)>=2,"advance count");uint64_t n=L*C;check(n<=raw.size()-in.at,"digits bounds");std::vector<int8_t> digits(n);memcpy(digits.data(),raw.data()+in.at,n);in.at+=n;
 AkitaStage2Config config{L,C,basis,UINT64_C(8)<<30,domain};check(akita_stage2_admit(&config,digits.data(),n)==AKITA_STAGE2_OK,"admit");
 auto invalid=config;invalid.initial_domain_len=3;check(akita_stage2_admit(&invalid,digits.data(),n)==AKITA_STAGE2_INVALID,"bad domain admission");Lease owner;check(akita_stage2_create(&config,digits.data(),n,&owner.p)==AKITA_STAGE2_OK && owner.p,"create");
 std::fill(digits.begin(),digits.end(),int8_t(127));digits.clear();digits.shrink_to_fit(); // owner must own copied bytes
 uint32_t wrong=0;std::thread thread([&]{wrong=akita_stage2_destroy(owner.p);});thread.join();check(wrong==AKITA_STAGE2_THREAD,"thread boundary");
 std::vector<uint8_t> actual;uint64_t current=C;
 for(uint64_t step=0;step<=advances;step++){
  Request req;req.read(in,L,current,step==0);AkitaStage2Message msg;memset(&msg,0x5A,sizeof(msg));auto before=msg;
  req.r.input_coefficients++;uint32_t bad=step==0?akita_stage2_compact_entry(owner.p,&req.r,&msg):akita_stage2_advance(owner.p,&req.r,&msg);req.r.input_coefficients--;
  check(bad==AKITA_STAGE2_INVALID && memcmp(&msg,&before,sizeof(msg))==0,"presubmit output/state");
  uint32_t status=step==0?akita_stage2_compact_entry(owner.p,&req.r,&msg):akita_stage2_advance(owner.p,&req.r,&msg);check(status==AKITA_STAGE2_OK,"native operation");
  append(actual,msg.ordinary,6);append(actual,msg.additional,4);current/=step==0?4:2;
  check(akita_stage2_compact_entry(owner.p,&req.r,&msg)==AKITA_STAGE2_STATE,"duplicate entry");
 }
 check(in.at==raw.size(),"trailing input");std::vector<AkitaStage2Ext> tail(L*current,{123,456});
 check(akita_stage2_finish(owner.p,tail.data(),tail.size()-1)==AKITA_STAGE2_INVALID,"finish count");for(auto x:tail)check(x.c0==123&&x.c1==456,"finish failure write");
 check(akita_stage2_finish(owner.p,tail.data(),tail.size())==AKITA_STAGE2_OK,"finish");append(actual,tail.data(),tail.size());
 check(akita_stage2_finish(owner.p,tail.data(),tail.size())==AKITA_STAGE2_STATE,"consumed finish");check(akita_stage2_destroy(owner.p)==AKITA_STAGE2_OK,"destroy");owner.p=nullptr;
 check(actual==expected,"optimized CPU expected mismatch");std::ofstream out(argv[3],std::ios::binary|std::ios::trunc);check(bool(out),"output open");out.write((char*)actual.data(),actual.size());out.close();check(bool(out),"output write");
 std::cout<<"OWNED_CONTINUOUS_PASS source_copied=true thread_checked=true finish_consumed=true\n";return 0;
}catch(const std::exception& e){std::cerr<<"owned fixture error: "<<e.what()<<"\n";return 1;}}
