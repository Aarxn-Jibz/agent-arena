import os
import unittest

from arena.secondary import CacheBenchmark, LogBenchmark, SearchBenchmark
from arena.sandbox import SandboxConfig

CACHE_C = '''#include <stdio.h>
#include <string.h>
int main(void){int cap,n,k,v,keys[256],vals[256],used=0;char op[8];
 if(scanf("%d %d",&cap,&n)!=2)return 1;
 for(int i=0;i<n;i++){if(scanf("%7s %d",op,&k)!=2)return 1;
  int j=0;while(j<used&&keys[j]!=k)j++;
  if(!strcmp(op,"GET")){if(j<used)printf("HIT %d\\n",vals[j]);else puts("MISS");}
  else {if(scanf("%d",&v)!=1)return 1;
   if(j==used&&used<cap){j=used++;keys[j]=k;puts("STORED");}
   else if(j==used){j=0;printf("EVICT %d\\n",keys[j]);keys[j]=k;}
   else puts("STORED");vals[j]=v;}}
 return 0;}'''

LOG_C = '''#include <stdio.h>
#include <string.h>
int main(int argc,char **argv){char line[256],stamp[32],level[32],src[32],msg[64];int n=0;
 if(argc!=4||strcmp(argv[1],"count"))return 1;
 while(fgets(line,sizeof line,stdin))if(sscanf(line,"%31[^|]|%31[^|]|%31[^|]|%63[^\\n]",stamp,level,src,msg)==4)
  if(!strcmp(level,argv[2]))n++;
 printf("%d\\n",n);return 0;}'''

SEARCH_C = '''#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv){if(argc<2)return 1;
 if(!strcmp(argv[1],"build")){int c;while((c=getchar())!=EOF)putchar(c);return 0;}
 if(argc!=3||strcmp(argv[1],"query"))return 1;
 unsigned char b[4];if(fread(b,1,4,stdin)!=4)return 1;
 unsigned n=((unsigned)b[0]<<24)|((unsigned)b[1]<<16)|((unsigned)b[2]<<8)|b[3];
 char *text=malloc(n+1),query[64],*outer,*inner,*line,*word; if(!text)return 1;
 if(fread(text,1,n,stdin)!=n||scanf("%63s",query)!=1)return 1;text[n]=0;
 line=strtok_r(text,"\\n",&outer);line=strtok_r(NULL,"\\n",&outer);
 int id=0,first=1;while(line){int found=0;word=strtok_r(line," ",&inner);
  while(word){if((!strcmp(argv[2],"exact")&&!strcmp(word,query))||
   (!strcmp(argv[2],"prefix")&&!strncmp(word,query,strlen(query))))found=1;
   word=strtok_r(NULL," ",&inner);}
  if(found){if(!first)putchar(' ');printf("%d",id);first=0;}id++;line=strtok_r(NULL,"\\n",&outer);}
 putchar('\\n');free(text);return 0;}'''


class SecondaryTest(unittest.TestCase):
    def test_cache_allows_any_reported_eviction(self):
        trace = [("PUT", 1, 10), ("PUT", 2, 20), ("PUT", 3, 30),
                 ("GET", 1, None), ("GET", 2, None), ("GET", 3, None)]
        self.assertEqual(CacheBenchmark.check(trace, ["STORED", "STORED", "EVICT 1",
                                                      "MISS", "HIT 20", "HIT 30"], 2),
                         (True, 2, 1))
        self.assertFalse(CacheBenchmark.check(trace, ["STORED"] * 6, 2)[0])
        b = CacheBenchmark()
        challenge = b.initialize(3)
        self.assertEqual(b.generate_trace(challenge), b.generate_trace(challenge))
        self.assertFalse(b.validate_challenge({**challenge, "capacity": 0})[0])

    def test_log_oracle_and_generation(self):
        b = LogBenchmark()
        challenge = {**b.initialize(2), "records": 4, "malformed": 2}
        rows = b.generate(challenge)
        self.assertEqual(rows, b.generate(challenge))
        self.assertEqual(len(rows), 6)
        self.assertTrue(b.oracle(rows, challenge).endswith(b"\n"))
        self.assertTrue(b.oracle(rows, {**challenge, "query": "by_level"}).startswith(b"INFO "))

    def test_search_oracle_and_generation(self):
        b = SearchBenchmark()
        challenge = b.initialize(5)
        self.assertEqual(b.generate(challenge), b.generate(challenge))
        self.assertEqual(b.oracle([["red", "blue"], ["blue"], ["red"]], "red", "exact"), b"0 2\n")
        self.assertEqual(b.oracle([["red", "blue"], ["blue"], ["red"]], "re", "prefix"), b"0 2\n")

    @unittest.skipUnless(os.environ.get("RUN_DOCKER_SANDBOX_TESTS") == "1", "Docker required")
    def test_judges_reject_wrong_candidate_in_docker(self):
        source = "int main(void){return 0;}"
        config = SandboxConfig(output_bytes=4096)
        for benchmark in (CacheBenchmark(), LogBenchmark(), SearchBenchmark()):
            challenge = benchmark.initialize(2)
            if benchmark.name == "search":
                challenge["queries"] = 2
            result = benchmark.evaluate(source, challenge, config)
            self.assertFalse(result["accepted"], benchmark.name)

    @unittest.skipUnless(os.environ.get("RUN_DOCKER_SANDBOX_TESTS") == "1", "Docker required")
    def test_positive_docker_contracts(self):
        config = SandboxConfig(output_bytes=4096)
        cache = CacheBenchmark()
        cache_case = {**cache.initialize(1), "capacity": 4, "keys": 4, "operations": 8}
        self.assertTrue(cache.evaluate(CACHE_C, cache_case, config)["accepted"])
        log = LogBenchmark()
        log_case = {**log.initialize(1), "records": 8, "malformed": 0}
        self.assertTrue(log.evaluate(LOG_C, log_case, config)["accepted"])
        search = SearchBenchmark()
        search_case = {**search.initialize(1), "documents": 4, "words_per_doc": 3,
                       "queries": 2}
        self.assertTrue(search.evaluate(SEARCH_C, search_case, config)["accepted"])
