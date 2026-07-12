/*
 * msp430_biosignal_acquisition.c
 *
 * EMG/EOG biosignal acquisition firmware for MSP430.
 * Samples 3 analog channels (2x EMG, 1x EOG) at 1kHz via ADC12,
 * packages the readings into a 9-byte frame, and streams them
 * over UART at 115200 bps.
 *
 * Packet format (9 bytes):
 *   [0] Header  0xAA
 *   [1] Header  0x77
 *   [2] Header  0xAA
 *   [3] EMG0 high nibble (upper 4 bits)
 *   [4] EMG0 low byte
 *   [5] EMG1 high nibble
 *   [6] EMG1 low byte
 *   [7] EOG  high nibble
 *   [8] EOG  low byte
 */

#include <msp430.h>

// ===== Global Variables =====
unsigned short adc0, adc1, adc2;   // ADC readings in mV (0~2500)
unsigned char Packet[9];           // UART packet buffer

// ===== Function Declarations =====
void ReadAdc12(void);

// ===== Main =====
void main(void)
{
    unsigned int i;

    // ----- Stop watchdog timer -----
    WDTCTL = WDTPW + WDTHOLD;

    // ----- Clock setup (6 MHz) -----
    BCSCTL1 &= ~XT2OFF;               // Enable XT2
    do {
        IFG1 &= ~OFIFG;               // Clear oscillator fault flag
        for (i = 0; i < 0xFF; i++);   // Stabilization delay
    } while ((IFG1 & OFIFG));         // Wait until stable

    BCSCTL2 |= SELM_2;                // MCLK = XT2CLK = 6MHz
    BCSCTL2 |= SELS;                  // SMCLK = XT2CLK = 6MHz

    // ----- Port setup -----
    P3SEL = BIT4 | BIT5;              // P3.4 = UART TX, P3.5 = UART RX
    P6SEL |= BIT0 | BIT1 | BIT3;      // P6.0(A0), P6.1(A1), P6.3(A3) = ADC input
    P6DIR &= ~(BIT0 | BIT1 | BIT3);   // Set as input
    P6OUT = 0x00;

    // ----- UART setup (115200 bps) -----
    ME1 |= UTXE0 + URXE0;             // Enable UART TX/RX
    UCTL0 |= CHAR;                    // 8-bit character
    UTCTL0 |= SSEL0 | SSEL1;          // Clock source: SMCLK
    UBR00 = 0x34;                     // 6MHz / 52 ~= 115200 bps
    UBR10 = 0x00;
    UMCTL0 = 0x00;                    // No modulation
    UCTL0 &= ~SWRST;                  // Initialize UART

    // ----- ADC12 setup -----
    ADC12CTL0 = ADC12ON | REFON | REF2_5V;   // ADC on, 2.5V reference
    ADC12CTL0 |= MSC;                        // Multiple sample and conversion
    ADC12CTL1 = ADC12SSEL_3 | ADC12DIV_7 | CONSEQ_1;  // SMCLK/8, sequence mode
    ADC12CTL1 |= SHP;                        // Sample-and-hold pulse mode

    // Channel mapping
    ADC12MCTL0 = SREF_0 | INCH_0;            // MEM0 <- A0 (P6.0)
    ADC12MCTL1 = SREF_0 | INCH_1;            // MEM1 <- A1 (P6.1)
    ADC12MCTL2 = SREF_0 | INCH_3 | EOS;      // MEM2 <- A3 (P6.3), end of sequence

    ADC12CTL0 |= ENC;                        // Enable conversion

    // ----- Timer A setup (1 kHz sampling) -----
    TACTL = TASSEL_2 + MC_1;          // SMCLK, up mode
    TACCTL0 = CCIE;                   // Enable CCR0 interrupt
    TACCR0 = 6000;                    // 6MHz / 6000 = 1kHz (1ms period)

    // ----- Enter low power mode, wait for interrupts -----
    _BIS_SR(LPM0_bits + GIE);
}

// ===== Timer A0 ISR (fires every 1 ms) =====
#pragma vector = TIMERA0_VECTOR
__interrupt void TimerA0_interrupt()
{
    int j;

    // Read and convert ADC values
    ReadAdc12();

    // ----- Build packet -----
    // Header (3 bytes)
    Packet[0] = 0xAA;
    Packet[1] = 0x77;
    Packet[2] = 0xAA;

    // EMG channel 0 (2 bytes)
    Packet[3] = (adc0 >> 8) & 0x0F;   // Upper 4 bits
    Packet[4] = adc0 & 0xFF;          // Lower 8 bits

    // EMG channel 1 (2 bytes)
    Packet[5] = (adc1 >> 8) & 0x0F;
    Packet[6] = adc1 & 0xFF;

    // EOG channel (2 bytes)
    Packet[7] = (adc2 >> 8) & 0x0F;
    Packet[8] = adc2 & 0xFF;

    // ----- Transmit over UART -----
    for (j = 0; j < 9; j++) {
        while (!(IFG1 & UTXIFG0));   // Wait until TX buffer is ready
        TXBUF0 = Packet[j];
    }
}

// ===== ADC read/convert =====
void ReadAdc12(void)
{
    unsigned short raw0, raw1, raw2;

    // Read raw ADC values (0~4095)
    raw0 = ADC12MEM0;
    raw1 = ADC12MEM1;
    raw2 = ADC12MEM2;

    // Convert to millivolts (2.5V reference)
    // voltage_mV = (ADC_value * 2500) / 4096
    adc0 = (unsigned short)((unsigned long)raw0 * 2500 / 4096);
    adc1 = (unsigned short)((unsigned long)raw1 * 2500 / 4096);
    adc2 = (unsigned short)((unsigned long)raw2 * 2500 / 4096);

    // Trigger next conversion
    ADC12CTL0 |= ADC12SC;
}
