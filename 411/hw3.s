/* hw3.s -*- Daniel Choo -*- */
.text
.global _start

_start:
  mov r0, #0               /* i counter */
	ldr r1,=A                /* Load in array A */
  ldr r2,=B	               /* Load in array B */
	mov r3, #0               /* Offset counter (I). */
	mov r4, #0               /* Keep track of D's location */
	mov r7, #0               /* Temp */
	mov r8, #0               /* Offset counter (J). */
	ldr r9,=C                /* Load in C */
	ldr r10,=D               /* Load in D */
	mov r11, #0              /* j counter */
	
c_loop:
	ldr r5, [r1, r3]          /* Load A[i] */
	ldr r6, [r2, r3]          /* Load B[i] */
	
  d_loop:		
		ldr r12, [r1, r8]       /* Load in A[j] */
		mul r7, r6, r5          /* temp = B[i] * A[j] */
		mov [r10, r4], r7       /* Insert temp into D*/
		add r8, #4              /* Add 4 to the offset counter.*/
		add r4, #4              /* Add 4 to D counter. */           
		mov r7, #0              /* Reset temp */
		cmp r11, #4             /* Make sure we're still in-bounds. */
		blt d_loop							/* If we are, branch back into d_loop */
	
	mul r7, r5, r6            /* Multiply A[i] * B[i]  */
	add r9, r7                /* Add the product to C */
	add r0, #1                /* Increment by 1; counter */
	add r3, #4                /* Increment by 4; offset */	
	mov r7, #0	              /* Set temp to 0 */
	mov r8, #0                /* Reset j for the next iteration */
	
  cmp r0, #4                /* Compare counter to 0 */
	blt c_loop                /* If less than 4, branch back into c_loop */	
	
.data

size: .word 4
A:  .word 1, 2, 3, 4
B:  .word 9, 8, 7, 6
C:	.word 0
D:  .word 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0

	
	
	
