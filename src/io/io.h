/*! PArticle Wake ANalysis
 * \file io.h
 * \brief Header for IO class
 *
 * @author Puneet Singh
 * @date 04/01/2021
 */
#ifndef IO_H_
#define IO_H_

#include <stdio.h>
#include <iostream>
#include "src/utils/print_utils.h"

namespace pawan {

class __io {
   private:
    std::string _root; /*!< Root directory */
    std::string _name; /*!< File name */

   public:
    //! Constructor
    /*
     * Creates empty directory
     */
    __io();

    //! Destructor
    /*
     * Deletes particles
     */
    ~__io() = default;

    //! Print all I/O data
    /*
     * Print root directory and filename
     */
    virtual void print();

    //! Get filename
    /*
     * Returns filename
     */
    virtual std::string getFile();

    //! Create binary file
    /*
     * Creates and return binary file with filename
     */
    virtual FILE* create_binary_file(std::string suffix);

    //! Open binary file to read
    /*
     * Opns binary file with filename
     */
    virtual FILE* open_binary_file(std::string suffix);
};
}  // namespace pawan
#endif
