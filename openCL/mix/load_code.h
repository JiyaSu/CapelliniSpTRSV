#ifndef LOAD_CODE_H
#define LOAD_CODE_H

bool LoadSourceFromFile(const char* filename,
                        char* & sourceCode )
{
    bool error = false;
    FILE* fp = NULL;
    int nsize = 0;
    
    // Open the shader file
    
    fp = fopen(filename, "rb");
    if( !fp )
    {
        error = true;
    }
    else
    {
        // Allocate a buffer for the file contents
        fseek( fp, 0, SEEK_END );
        nsize = ftell( fp );
        fseek( fp, 0, SEEK_SET );
        
        sourceCode = (char *)malloc((nsize + 1) * sizeof(char));
        //sourceCode = new char [ nsize + 1 ];
        if( sourceCode )
        {
            fread( sourceCode, 1, nsize, fp );
            sourceCode[ nsize ] = 0; // Don't forget the NULL terminator
        }
        else
        {
            error = true;
        }
        
        fclose( fp );
    }
    
    return error;
}
#endif
