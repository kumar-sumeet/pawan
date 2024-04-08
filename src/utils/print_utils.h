/*! PArticle Wake ANalysis
 * @file print_utils.h
 * @brief Print utilities for PAWAN
 *
 * @author Puneet Singh
 * @date 04/01/2021
 *
 */

#ifndef PRINT_UTILS_H_
#define PRINT_UTILS_H_

const bool DEBUG_PRINT=false;

#include <iostream>
#include <sstream>
#include <string>

/*! \fn inline void HEADER(std::ostream &os = std::cout)
 * \brief Print string
 */
inline void PAWAN(std::ostream &os = std::cout){
	os << "********************************************************************************" << std::endl;
	os << "\t\t\t    _"<< std::endl;
	os << "\t\t    o o o  |_|  /\\ \\  /\\  / /\\  |\\ |  o o o"<< std::endl;
	os << "\t\t     o o o |   /  \\ \\/  \\/ /  \\ | \\| o o o" << std::endl;
	os << std::endl;
	os << "\t\t\t     PArticle Wake ANalysis" << std::endl;
	os << "\t\t\t     (c) Puneet Singh 2021" << std::endl;
	os << "********************************************************************************" << std::endl;
};

inline void DOUT(std::string s, std::ostream &os = std::cout){
    if (DEBUG_PRINT) {
        os << s << std::endl;
    }
};


/*! \fn inline void OUT(std::string s, std::ostream &os = std::cout)
 * \brief Print string
 * \param s	String
 * \param os	Output stream
 */
inline void OUT(std::string s, std::ostream &os = std::cout){
	os << s << std::endl;
};

/*! \fn inline void OUT(std::string s, size_t v, std::ostream &os = std::cout)
 * \brief Print string and size
 * \param s	String
 * \param v	Output quantity
 * \param os	Output stream
 */
inline void OUT(std::string s, size_t v, std::ostream &os = std::cout){
	os << s << " = "<< v << std::endl;
};

/*! \fn inline void OUT(std::string s, double d, std::ostream &os = std::cout)
 * \brief Print string and size
 * \param s	String
 * \param d	Output quantity
 * \param os	Output stream
 */
inline void OUT(std::string s, double d, std::ostream &os = std::cout){
	os << s << " = "<< d << std::endl;
};

/*! \fn inline void OUT(std::string s, T v, std::ostream &os = std::cout)
 * \brief Print string and value
 * \param s	String
 * \param v	Output quantity
 * \param os	Output stream
 */
//template <typename T>
//inline void OUT(std::string s, T v, std::ostream &os = std::cout){
	//os << s << " = "<< v << std::endl;
//};

#endif
