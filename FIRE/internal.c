#include<stdio.h>
#include<conio.h>
#include<string.h>
#include<stdlib.h>

void main()
{
    char str[]="HELLO WORLD";
    int i,len;
    char str1[11];
    len=strlen(str);
    for(i=0;i<len;i++)
    {
        str1[i]=str[i]^0;
        printf("a%d = %c /n",i,str1[i]);

    }
    printf("/n");
}