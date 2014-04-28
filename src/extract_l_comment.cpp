#include <sstream>
#include <fstream>
#include <string>
#include <iostream>
#include <vector>

bool importCSV(const char* path, const char* tablename, int start = 0, int limit = 0, bool force_dict=false) {
        std::string line;

        //build filenames
        std::string h_filename = path;
        h_filename.append("/");
        h_filename.append(tablename);
        std::string v_basefilename = h_filename;
        h_filename.append("_header.csv");
        std::vector< std::string > v_filenames;

        //find tbl files and number of lines to import first
        std::string v_filename = v_basefilename+".tbl";
        unsigned size=0;
        for(unsigned i=1; true; ++i) {
            std::ifstream value_file(v_filename.c_str());
            if(value_file.good()) {
                while(getline(value_file, line) && (limit==0 || size<limit))
                    ++size;
                value_file.close();
                v_filenames.push_back(v_filename);
                v_filename=v_basefilename;
                v_filename.append(".tbl.");
                std::stringstream t;
                t<<i;
                v_filename.append(t.str());
            } else {
                if(i==1) {
                    v_filename=v_basefilename;
                    v_filename.append(".tbl.1");
                } else {
                    break;
                }
            }
        }

        if(!size)
            return false;

        //get column data
        std::ifstream header_file(h_filename.c_str());
        std::getline(header_file,line);

        if(line[line.size()-1]=='\r')
            line.resize(line.size()-1);
        std::stringstream colTitleStream(line);

        std::getline(header_file,line);

        if(line[line.size()-1]=='\r')
            line.resize(line.size()-1);
        std::stringstream colTypeStream(line);

        std::string titleCell;
        std::string typeCell;

        unsigned starting_col = 0;
        unsigned reserve_size=limit ? limit : size;

        std::cout<<"Importing "<<reserve_size<<" rows"<<std::endl;
//        for(int i = 0;std::getline(colTitleStream,titleCell,'|');++i)    {
//            std::transform(titleCell.begin(),titleCell.end(),titleCell.begin(),::toupper);
//        }
        std::vector<std::string> L_COMMENT;
        L_COMMENT.reserve(reserve_size);

        //import tbl files
        unsigned row = 0;
        std::string buffer;
        for(std::vector< std::string >::iterator f_it = v_filenames.begin(); f_it!=v_filenames.end(); ++f_it) {
            unsigned parse_end=limit ? (limit-row) : size;
            unsigned parse_start=start<row ? 0 : (start-row);

            std::ifstream value_file(f_it->c_str());
            while (std::getline(value_file,buffer))  {
            	buffer.resize(buffer.size()-1);
            	int beginning = buffer.find_last_of('|');
            	L_COMMENT.push_back(buffer.substr(beginning+1));
            	++row;
			}

            value_file.close();
            std::cout<<*f_it<<": "<<row<<"..."<<std::endl;
            std::cout.flush();
            if(row>=reserve_size) {
                break;
            }
        }
        std::cout<<"Writing"<<std::endl;

        std::ofstream final_file("L_COMMENT");
        if(final_file.good()) {
        	for(std::vector< std::string >::iterator it = L_COMMENT.begin();it!=L_COMMENT.end();++it) {
        		final_file<<*it<<std::endl;
        	}
            return true;
        }
        return false;
    }


int main(int ac, char** av) {
	importCSV("/home/hannes/tpch/","lineitem");
}
