package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	mat "github.com/klahssen/go-mat"
	"github.com/klahssen/nn"
)

func main() {
	f := flag.String("f", "", "filepath to json config")
	flag.Parse()
	p := nn.Perceptron{}
	if err := p.FromJSON(*f); err != nil {
		fmt.Fprintf(os.Stderr, "failed to construct perceptron from config: %s\n", err.Error())
		os.Exit(1)
	}
	//reader := bufio.NewReader(os.Stdin)
	scanner := bufio.NewScanner(os.Stdin)
	vals := []string{}
	inSize := p.Size()
	for {
		fmt.Printf("give me %d values to challenge me:\n", inSize)
		scanner.Scan()
		vals = strings.Split(scanner.Text(), " ")
		if len(vals) != inSize {
			fmt.Printf("expected %d values received %d\n", inSize, len(vals))
			continue
		}
		x := make([]float64, len(vals))
		for i := range vals {
			f, err := strconv.ParseFloat(vals[i], 64)
			if err != nil {
				fmt.Printf("error: %s\n", err.Error())
				continue
			}
			x[i] = f
		}
		res, err := p.Compute(mat.NewM64(inSize, 1, x))
		if err != nil {
			fmt.Printf("error: %s\n", err.Error())
			continue
		}
		fmt.Printf("I guess the result is %.2f\n", res)
	}
}
