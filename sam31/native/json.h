#pragma once
// Small, bounded JSON reader/writer for the GGUF tensor map and native reports.
// No external interpreter or JSON dependency is involved in inference.
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>
namespace sam31 {
    namespace native {
        class Json {
            public: using object = std::map<std::string, Json>;
            using array = std::vector<Json>;
            using value = std::variant<std::nullptr_t, bool, double, std::string, array, object>;
            value v = nullptr;
            Json() = default;
            Json(bool x):v(x) {
            }
            Json(int x):v(double(x)) {
            }
            Json(size_t x):v(double(x)) {
            }
            Json(double x):v(x) {
            }
            Json(const char*x):v(std::string(x)) {
            }
            Json(std::string x):v(std::move(x)) {
            }
            Json(array x):v(std::move(x)) {
            }
            Json(object x):v(std::move(x)) {
            }
            const object& obj() const {
                if(!std::holds_alternative<object>(v))fail("expected object");
                return std::get<object>(v);
            }
            const array& arr() const {
                if(!std::holds_alternative<array>(v))fail("expected array");
                return std::get<array>(v);
            }
            const std::string& str() const {
                if(!std::holds_alternative<std::string>(v))fail("expected string");
                return std::get<std::string>(v);
            }
            double number() const {
                if(!std::holds_alternative<double>(v))fail("expected number");
                return std::get<double>(v);
            }
            size_t integer() const {
                double x=number();
                if(x<0||x>9007199254740991.0||std::floor(x)!=x)fail("expected bounded unsigned integer");
                return size_t(x);
            }
            bool boolean() const {
                if(!std::holds_alternative<bool>(v))fail("expected boolean");
                return std::get<bool>(v);
            }
            const Json& at(const std::string&key) const {
                const auto &o=obj();
                auto it=o.find(key);
                if(it==o.end())fail("missing key: "+key);
                return it->second;
            }
            bool has(const std::string&key) const {
                return obj().count(key)!=0;
            }
            Json& operator[](const std::string &key) {
                if(std::holds_alternative<std::nullptr_t>(v))v=object {
                };
                return std::get<object>(v)[key];
            }
            [[noreturn]] static void fail(const std::string&s) {
                throw std::runtime_error("JSON: "+s);
            }
            static std::string quote(const std::string&s) {
                std::ostringstream o;
                o<<'"';
                for(unsigned char c:s) {
                    switch(c) {
                        case '"':o<<"\\\"";
                        break;
                        case '\\':o<<"\\\\";
                        break;
                        case '\n':o<<"\\n";
                        break;
                        case '\r':o<<"\\r";
                        break;
                        case '\t':o<<"\\t";
                        break;
                        default:if(c<32)o<<"\\u00"<<std::hex<<std::setw(2)<<std::setfill('0')<<int(c)<<std::dec;
                        else o<<c;
                    }
                }
                o<<'"';
                return o.str();
            }
            std::string dump() const {
                if(std::holds_alternative<std::nullptr_t>(v))return "null";
                if(auto p=std::get_if<bool>(&v))return *p?"true":"false";
                if(auto p=std::get_if<double>(&v)) {
                    if(!std::isfinite(*p))fail("non-finite output");
                    std::ostringstream o;
                    o<<std::setprecision(17)<<*p;
                    return o.str();
                }
                if(auto p=std::get_if<std::string>(&v))return quote(*p);
                std::string s;
                bool first=true;
                if(auto p=std::get_if<array>(&v)) {
                    s="[";
                    for(const auto&x:*p) {
                        if(!first)s+=",";
                        first=false;
                        s+=x.dump();
                    }
                    return s+"]";
                }
                s="{";
                for(const auto&kv:obj()) {
                    if(!first)s+=",";
                    first=false;
                    s+=quote(kv.first)+":"+kv.second.dump();
                }
                return s+"}";
            }
            static Json parse(const std::string&text) {
                if(text.size()>64u*1024u*1024u)fail("input exceeds 64 MiB");
                struct Parser {
                    const std::string&s;
                    size_t p=0,nodes=0;
                    void ws() {
                        while(p<s.size()&&(s[p]==' '||s[p]=='\t'||s[p]=='\n'||s[p]=='\r'))++p;
                    }
                    char get() {
                        if(p==s.size())fail("unexpected end");
                        return s[p++];
                    }
                    bool take(char c) {
                        ws();
                        if(p<s.size()&&s[p]==c) {
                            ++p;
                            return true;
                        }
                        return false;
                    }
                    unsigned hex4() {
                        unsigned n=0;
                        for(int i=0; i<4; ++i) {
                            char c=get();
                            unsigned a;
                            if(c>='0'&&c<='9')a=c-'0';
                            else if(c>='a'&&c<='f')a=c-'a'+10;
                            else if(c>='A'&&c<='F')a=c-'A'+10;
                            else fail("invalid unicode escape");
                            n=n*16+a;
                        }
                        return n;
                    }
                    void utf8(std::string&o,unsigned c) {
                        if(c<0x80)o+=char(c);
                        else if(c<0x800) {
                            o+=char(0xc0|(c>>6));
                            o+=char(0x80|(c&63));
                        }
                        else if(c<0x10000) {
                            o+=char(0xe0|(c>>12));
                            o+=char(0x80|((c>>6)&63));
                            o+=char(0x80|(c&63));
                        }
                        else {
                            o+=char(0xf0|(c>>18));
                            o+=char(0x80|((c>>12)&63));
                            o+=char(0x80|((c>>6)&63));
                            o+=char(0x80|(c&63));
                        }
                    }
                    std::string string() {
                        ws();
                        if(get()!='"')fail("expected quote");
                        std::string o;
                        while(true) {
                            unsigned char c=get();
                            if(c=='"')return o;
                            if(c<32)fail("unescaped control");
                            if(c!='\\') {
                                o+=char(c);
                                continue;
                            }
                            switch(get()) {
                                case '"':o+='"';
                                break;
                                case '\\':o+='\\';
                                break;
                                case '/':o+='/';
                                break;
                                case 'b':o+='\b';
                                break;
                                case 'f':o+='\f';
                                break;
                                case 'n':o+='\n';
                                break;
                                case 'r':o+='\r';
                                break;
                                case 't':o+='\t';
                                break;
                                case 'u': {
                                    unsigned x=hex4();
                                    if(x>=0xd800&&x<=0xdbff) {
                                        if(get()!='\\'||get()!='u')fail("missing low surrogate");
                                        unsigned y=hex4();
                                        if(y<0xdc00||y>0xdfff)fail("invalid low surrogate");
                                        x=0x10000+((x-0xd800)<<10)+(y-0xdc00);
                                    }
                                    else if(x>=0xdc00&&x<=0xdfff)fail("unpaired surrogate");
                                    utf8(o,x);
                                    break;
                                }
                                default:fail("unknown escape");
                            }
                        }
                    }
                    Json read(size_t depth=0) {
                        ws();
                        if(depth>32||++nodes>2000000)fail("structural limit");
                        if(p==s.size())fail("empty value");
                        char c=s[p];
                        if(c=='"')return string();
                        if(c=='{') {
                            ++p;
                            object o;
                            if(take('}'))return o;
                            do {
                                auto k=string();
                                if(!take(':'))fail("expected colon");
                                auto val=read(depth+1);
                                if(!o.emplace(k,std::move(val)).second)fail("duplicate key: "+k);
                            }
                            while(take(','));
                            if(!take('}'))fail("expected }");
                            return o;
                        }
                        if(c=='[') {
                            ++p;
                            array a;
                            if(take(']'))return a;
                            do {
                                a.push_back(read(depth+1));
                            }
                            while(take(','));
                            if(!take(']'))fail("expected ]");
                            return a;
                        }
                        for(auto lit: {
                            "true","false","null"
                        }
                        ) {
                            std::string l=lit;
                            if(s.compare(p,l.size(),l)==0) {
                                p+=l.size();
                                if(l=="true")return true;
                                if(l=="false")return false;
                                return Json {
                                };
                            }
                        }
                        size_t b=p;
                        if(s[p]=='-')++p;
                        if(p==s.size())fail("invalid number");
                        if(s[p]=='0')++p;
                        else {
                            if(s[p]<'1'||s[p]>'9')fail("invalid number");
                            while(p<s.size()&&s[p]>='0'&&s[p]<='9')++p;
                        }
                        if(p<s.size()&&s[p]=='.') {
                            ++p;
                            size_t q=p;
                            while(p<s.size()&&s[p]>='0'&&s[p]<='9')++p;
                            if(p==q)fail("invalid fraction");
                        }
                        if(p<s.size()&&(s[p]=='e'||s[p]=='E')) {
                            ++p;
                            if(p<s.size()&&(s[p]=='+'||s[p]=='-'))++p;
                            size_t q=p;
                            while(p<s.size()&&s[p]>='0'&&s[p]<='9')++p;
                            if(p==q)fail("invalid exponent");
                        }
                        double x=std::stod(s.substr(b,p-b));
                        if(!std::isfinite(x))fail("non-finite number");
                        return x;
                    }
                }
                parser {
                    text
                };
                auto result=parser.read();
                parser.ws();
                if(parser.p!=text.size())fail("trailing data");
                return result;
            }
        };
    }
}
